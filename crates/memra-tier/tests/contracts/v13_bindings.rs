use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::{cell::RefCell, rc::Rc};

// Existing DeviceOwner + Governor fake, with a separately retained resource cell.
// release() removes its registered cell handle; take moves the original backing.
struct HandBack {
    owner: DeviceOwner,
    lease: DeviceLease,
    pins: Vec<DeviceLease>,
    backing: Rc<RefCell<Option<Vec<u8>>>>,
    original_pointer: usize,
    gov: Shared,
    charge: ChargedLease,
    unknown: bool,
    taken: bool,
}
impl DeviceHandBackFixture for HandBack {
    type Backing = Vec<u8>;
    fn take(&mut self) -> Result<Vec<u8>> {
        if self.taken {
            return Err(Error::ForeignLease);
        }
        if self.unknown {
            return Err(Error::Quarantined);
        }
        self.owner.release(&self.lease)?;
        self.gov.borrow_mut().release(&self.charge)?;
        self.taken = true;
        Ok(self.backing.borrow_mut().take().unwrap())
    }
    fn advance(&mut self, step: HandBackStep) {
        match step {
            HandBackStep::Unknown => self.unknown = true,
            HandBackStep::Recover => self.unknown = false,
            HandBackStep::Producer | HandBackStep::Consumer | HandBackStep::Graph => {
                self.pins.pop().unwrap();
            }
        }
    }
    fn assert_registered_and_charged(&mut self) {
        assert_eq!(
            self.gov.borrow_mut().release(&self.charge),
            Err(Error::Busy)
        );
        assert_eq!(self.gov.borrow().used.device[0], 4);
        assert!(
            self.owner
                .resolve::<Rc<RefCell<Option<Vec<u8>>>>>(&self.lease)
                .is_ok()
        );
        assert_eq!(
            self.backing.borrow().as_ref().unwrap().as_ptr() as usize,
            self.original_pointer
        );
    }
    fn assert_returned_once(&mut self, backing: Vec<u8>) {
        assert_eq!(backing, bytes(4));
        assert_eq!(backing.as_ptr() as usize, self.original_pointer); // no copy/free
        assert!(
            self.owner
                .resolve::<Rc<RefCell<Option<Vec<u8>>>>>(&self.lease)
                .is_err()
        );
        assert_eq!(self.gov.borrow().used, TierBudget::zero(2));
        assert_eq!(
            self.gov.borrow_mut().release(&self.charge),
            Err(Error::AlreadyReleased)
        );
        assert!(self.backing.borrow().is_none());
    }
    fn reject_foreign(&mut self) {
        let mut foreign = DeviceOwner::new(0);
        assert_eq!(foreign.release(&self.lease), Err(Error::WrongOwner));
        self.assert_registered_and_charged();
    }
}
#[test]
fn original_device_backing_is_handed_back_only_after_last_pin() {
    let gov = shared();
    let mut req = request(0, Priority::MandatoryActive);
    req.bytes.device[0] = 4;
    let charge = gov.borrow_mut().reserve(&req).unwrap();
    let backing = Rc::new(RefCell::new(Some(bytes(4))));
    let original_pointer = backing.borrow().as_ref().unwrap().as_ptr() as usize;
    let mut owner = DeviceOwner::new(0);
    let lease = owner
        .register(31, 4, Box::new(backing.clone()), &charge)
        .unwrap();
    let pins = (0..3).map(|_| owner.retain(&lease).unwrap()).collect();
    device_hand_back(&mut HandBack {
        owner,
        lease,
        pins,
        backing,
        original_pointer,
        gov,
        charge,
        unknown: false,
        taken: false,
    });
}
