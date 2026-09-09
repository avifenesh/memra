use super::*;

/// The real `data.pkl` of the pinned Hebrew RNNT artifact, 126,903 bytes, carrying the whole
/// census and no weights. Extracted from `clean-step-21959.nemo`, archive sha256
/// 585c40f214a53e8b5ba565c6176aa6cb548c7c8ab2f7ec5dc6dd45dbdff46ee4.
const DATA_PKL: &[u8] =
    include_bytes!("../model_packs/nemotron_rnnt/fixtures/model_weights.data.pkl");

fn ustar(name: &str, size: usize, kind: u8) -> Vec<u8> {
    let mut header = vec![0u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    let octal = format!("{size:011o}\0");
    header[124..124 + octal.len()].copy_from_slice(octal.as_bytes());
    header[156] = kind;
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    header
}

fn pad(bytes: &mut Vec<u8>, payload: &[u8]) {
    bytes.extend_from_slice(payload);
    let tail = payload.len().div_ceil(512) * 512 - payload.len();
    bytes.extend(std::iter::repeat_n(0u8, tail));
}

#[test]
fn pax_extended_header_names_the_member_not_the_truncated_ustar_field() {
    let long = format!("./{}_tokenizer.model", "a".repeat(120));
    let payload = format!("path={long}\n");
    // The length prefix counts itself, the space and the newline, so it has a fixed point.
    let mut length = payload.len() + 2;
    while length.to_string().len() + 1 + payload.len() != length {
        length += 1;
    }
    let record = format!("{length} {payload}");
    let mut archive = Vec::new();
    archive.extend(ustar("././@PaxHeader", record.len(), b'x'));
    pad(&mut archive, record.as_bytes());
    archive.extend(ustar(&long[..100], 7, b'0'));
    pad(&mut archive, b"payload");
    archive.extend(vec![0u8; 1024]);

    let members = tar_members(&archive).unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, long);
    assert_eq!(members[0].size, 7);
    assert_eq!(
        &archive[members[0].offset..members[0].offset + 7],
        b"payload"
    );

    // Red arm: without the pax record the reader would hand back the 100-byte truncation.
    let mut truncated = Vec::new();
    truncated.extend(ustar(&long[..100], 7, b'0'));
    pad(&mut truncated, b"payload");
    assert_eq!(tar_members(&truncated).unwrap()[0].name, long[..100]);
}

#[test]
fn unsupported_tar_entries_and_broken_headers_are_refused() {
    let mut link = Vec::new();
    link.extend(ustar("./evil", 0, b'2'));
    assert!(matches!(tar_members(&link), Err(NemoError::Archive(_))));

    let mut garbage = vec![b'x'; 512];
    garbage[257..262].copy_from_slice(b"nope!");
    assert!(matches!(tar_members(&garbage), Err(NemoError::Archive(_))));

    let mut overrun = Vec::new();
    overrun.extend(ustar("./big", 4096, b'0'));
    assert!(matches!(tar_members(&overrun), Err(NemoError::Archive(_))));
}

#[test]
fn census_of_the_pinned_hebrew_rnnt_checkpoint() {
    let census = pickle_census(DATA_PKL).unwrap();
    assert_eq!(census.len(), 657);
    assert!(census.iter().all(|t| t.dtype == NemoDtype::F32));
    assert!(census.iter().all(NemoTensor::is_contiguous));
    assert_eq!(
        census.iter().map(NemoTensor::elements).sum::<u64>(),
        638_030_384
    );
    assert_eq!(
        census.iter().map(|t| t.elements() * 4).sum::<u64>(),
        2_552_121_536
    );

    // Checkpoint order is the contract order: the frontend buffers come first and the
    // prompt kernel closes the file.
    assert_eq!(census[0].name, "preprocessor.featurizer.window");
    assert_eq!(census[0].shape, vec![400]);
    assert_eq!(census[1].name, "preprocessor.featurizer.fb");
    assert_eq!(census[1].shape, vec![1, 128, 257]);
    assert_eq!(census[census.len() - 1].name, "prompt_kernel.2.bias");

    let group = |prefix: &str| census.iter().filter(|t| t.name.starts_with(prefix)).count();
    assert_eq!(group("preprocessor."), 2);
    assert_eq!(group("encoder."), 636);
    assert_eq!(group("decoder."), 9);
    assert_eq!(group("joint."), 6);
    assert_eq!(group("prompt_kernel."), 4);

    // Prompt conditioning is a real slot with real weights, not an id prepended by guesswork.
    let prompt: Vec<_> = census
        .iter()
        .filter(|t| t.name.starts_with("prompt_kernel."))
        .map(|t| (t.name.as_str(), t.shape.clone()))
        .collect();
    assert_eq!(
        prompt,
        vec![
            ("prompt_kernel.0.weight", vec![2048, 1152]),
            ("prompt_kernel.0.bias", vec![2048]),
            ("prompt_kernel.2.weight", vec![1024, 2048]),
            ("prompt_kernel.2.bias", vec![1024]),
        ]
    );

    // Both LSTM layers, four gates packed into every row, and the second layer takes the
    // first layer's 640-wide hidden state as its input.
    for layer in 0..2 {
        for gate in ["weight_ih", "weight_hh"] {
            let name = format!("decoder.prediction.dec_rnn.lstm.{gate}_l{layer}");
            let tensor = census.iter().find(|t| t.name == name).unwrap();
            assert_eq!(tensor.shape, vec![2560, 640], "{name}");
        }
    }
}

#[test]
fn every_storage_in_the_pinned_checkpoint_belongs_to_exactly_one_tensor() {
    let census = pickle_census(DATA_PKL).unwrap();
    let shared = shared_storage_groups(&census);
    // Measured, not assumed: this artifact aliases nothing, including the LSTM gates, so a
    // loader that silently substituted an aliased storage would not be caught by this file.
    // The detector stays because the upstream family is documented to share LSTM storage.
    assert!(shared.is_empty(), "unexpected aliasing: {shared:?}");
    assert!(census.iter().all(|t| t.storage_offset == 0));
    assert!(
        census.iter().all(|t| t.storage_elements == t.elements()),
        "a storage is larger than the tensor that reads it"
    );
}

#[test]
fn aliased_storages_are_reported_together() {
    let tensor = |name: &str, key: &str, offset: u64, extent: u64| NemoTensor {
        name: name.into(),
        dtype: NemoDtype::F32,
        shape: vec![extent],
        stride: vec![1],
        storage_offset: offset,
        storage_key: key.into(),
        storage_elements: 8,
    };
    let census = vec![
        tensor("lstm.weight_ih_l0", "7", 0, 4),
        tensor("lstm.weight_hh_l0", "7", 4, 4),
        tensor("joint.enc.bias", "8", 0, 8),
    ];
    let shared = shared_storage_groups(&census);
    assert_eq!(shared.len(), 1);
    assert_eq!(
        shared.get("7").unwrap(),
        &vec![
            "lstm.weight_ih_l0".to_owned(),
            "lstm.weight_hh_l0".to_owned()
        ]
    );
}

#[test]
fn a_checkpoint_that_names_an_executable_global_is_refused() {
    let mut pickle = vec![0x80, 0x02];
    pickle.extend_from_slice(b"cposix\nsystem\nq\x00.");
    let error = pickle_census(&pickle).unwrap_err();
    assert_eq!(
        error,
        NemoError::Pickle("checkpoint names disallowed global posix.system".into())
    );

    // The allowlist is not a prefix match either.
    let mut pickle = vec![0x80, 0x02];
    pickle.extend_from_slice(b"ctorch\nload\nq\x00.");
    assert!(matches!(pickle_census(&pickle), Err(NemoError::Pickle(_))));
}

#[test]
fn a_truncated_pickle_does_not_return_a_short_census() {
    let error = pickle_census(&DATA_PKL[..DATA_PKL.len() / 2]).unwrap_err();
    assert!(matches!(error, NemoError::Pickle(_)));
}

#[test]
fn zip_entries_refuse_compression_and_bad_offsets() {
    // Minimal stored zip with one entry, then the same zip flipped to deflate.
    let name = b"model_weights/data.pkl";
    let payload = b"hello";
    let mut zip = Vec::new();
    zip.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
    zip.extend_from_slice(&[0u8; 22]);
    zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
    zip.extend_from_slice(&0u16.to_le_bytes());
    zip.extend_from_slice(name);
    let data_at = zip.len();
    zip.extend_from_slice(payload);
    let central_at = zip.len();
    zip.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
    zip.extend_from_slice(&[0u8; 20]);
    zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
    zip.extend_from_slice(&[0u8; 12]);
    zip.extend_from_slice(&0u32.to_le_bytes());
    zip.extend_from_slice(name);
    let mut tail = Vec::new();
    tail.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
    tail.extend_from_slice(&[0u8; 6]);
    tail.extend_from_slice(&1u16.to_le_bytes());
    tail.extend_from_slice(&((zip.len() - central_at) as u32).to_le_bytes());
    tail.extend_from_slice(&(central_at as u32).to_le_bytes());
    tail.extend_from_slice(&0u16.to_le_bytes());
    zip.extend_from_slice(&tail);

    let entries = zip_entries(&zip).unwrap();
    assert_eq!(
        entries.get("model_weights/data.pkl"),
        Some(&(data_at, payload.len()))
    );

    let mut deflated = zip.clone();
    deflated[central_at + 10] = 8;
    assert!(matches!(zip_entries(&deflated), Err(NemoError::Zip(_))));
}
