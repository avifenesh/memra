use super::*;
use memra_tier::contracts::{BudgetRequest, Deadline, TierBudget};
use memra_tier::io::direct::{AlignedBuffer, AlignedFile};
use memra_tier::tier::Governor;

#[test]
fn common_governor_preserves_mandatory_pool_and_object_headroom() {
    let mut cap = TierBudget::zero(0);
    cap.pageable = 2 * 8191;
    cap.nvme = 32768;
    let mut headroom = TierBudget::zero(0);
    headroom.pageable = 8191;
    headroom.nvme = 16384;
    let g = std::rc::Rc::new(std::cell::RefCell::new(
        Governor::new(cap, headroom, 4, 0, Arc::new(|| 0)).unwrap(),
    ));
    let mut r = BudgetRequest {
        bytes: TierBudget::zero(0),
        priority: Priority::OptionalPrefetch,
        deadline: Deadline(100),
        tenant: [0; 32],
    };
    r.bytes.pageable = 8191;
    let optional = g.borrow_mut().reserve(&r).unwrap();
    let optional_pool = FakePinnedPool::new(1, 4096, 4096, 0, &optional).unwrap();
    assert!(matches!(g.borrow_mut().reserve(&r), Err(Error::Capacity)));
    r.priority = Priority::MandatoryActive;
    let mandatory = g.borrow_mut().reserve(&r).unwrap();
    let mandatory_pool = FakePinnedPool::new(1, 4096, 4096, 0, &mandatory).unwrap();
    assert!(matches!(g.borrow_mut().reserve(&r), Err(Error::Capacity)));
    assert_eq!(g.borrow_mut().release(&mandatory), Err(Error::Busy));
    drop(mandatory_pool);
    drop(optional_pool);
    g.borrow_mut().release(&mandatory).unwrap();
    g.borrow_mut().release(&optional).unwrap();
    let mut s = ExtentStore::new(MemoryBackend::default(), g.clone());
    let mut t = s.begin(key(), 264, Durability::Ephemeral).unwrap();
    s.put(&mut t, &payload(264)).unwrap();
    let m = s.commit(&mut t).unwrap();
    r.bytes.pageable = 0;
    r.bytes.nvme = 16384;
    r.priority = Priority::OptionalPrefetch;
    let opt_object = s.lease(&m, &r).unwrap();
    assert!(matches!(s.lease(&m, &r), Err(Error::Capacity)));
    r.priority = Priority::MandatoryActive;
    let active_object = s.lease(&m, &r).unwrap();
    assert!(matches!(s.lease(&m, &r), Err(Error::Capacity)));
    s.release(&active_object).unwrap();
    s.release(&opt_object).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(0));
}

#[test]
fn aligned_extent_io_rejects_misalignment_and_truncation() {
    let dir = OwnedDirectory::new();
    let path = dir.0.join("extent");
    std::fs::write(&path, vec![7; 8192]).unwrap();
    let file = AlignedFile::from_configured_file(
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap(),
    );
    let mut buffer = AlignedBuffer::new(8192).unwrap();
    assert_eq!(buffer.bytes().as_ptr() as usize % 4096, 0);
    assert_eq!(
        file.read_aligned(1, buffer.bytes_mut()),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        file.read_aligned(0, &mut buffer.bytes_mut()[1..]),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        file.write_aligned(1, buffer.bytes()),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        file.write_aligned(0, &buffer.bytes()[..4095]),
        Err(Error::InvalidLayout)
    );
    assert_eq!(file.read_aligned(0, buffer.bytes_mut()), Ok(8192));
    buffer.bytes_mut().fill(3);
    assert_eq!(file.write_aligned(0, buffer.bytes()), Ok(8192));
    file.sync_all().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), vec![3; 8192]);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(4097)
        .unwrap();
    assert_eq!(
        file.read_aligned(0, buffer.bytes_mut()),
        Err(Error::ShortIo {
            expected: 8192,
            actual: 4097
        })
    );
    assert!(matches!(AlignedBuffer::new(0), Err(Error::InvalidLayout)));
    assert!(matches!(
        AlignedBuffer::new(MAX_CHUNK + 8192),
        Err(Error::Capacity)
    ));
}

#[test]
fn direct_mode_is_explicit_and_never_silently_falls_back() {
    let dir = OwnedDirectory::new();
    let mut backend = FileBackend::open(&dir.0).unwrap();
    #[cfg(not(target_os = "linux"))]
    {
        assert_eq!(backend.set_direct_writes(true), Err(Error::Unsupported));
        assert!(matches!(
            AlignedFile::create_new(&dir.0.join("direct")),
            Err(Error::Unsupported)
        ));
    }
    #[cfg(target_os = "linux")]
    {
        backend.set_direct_writes(true).unwrap();
        backend.set_read_mode(memra_tier::io::direct::ReadMode::Uncached);
        let mut s = new_store(backend);
        let mut t = s.begin(key(), 4097, Durability::Ephemeral).unwrap();
        s.put(&mut t, &payload(4097)).unwrap();
        let m = s.commit(&mut t).unwrap();
        assert_eq!(read_manifest(&mut s, &m, 0).unwrap(), payload(4097));
        assert!(!std::fs::read_dir(&dir.0).unwrap().any(|p| {
            p.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".pending")
        }));
    }
}
