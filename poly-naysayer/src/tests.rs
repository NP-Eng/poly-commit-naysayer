use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::{Field, PrimeField};
use ark_poly::{MultilinearExtension, Polynomial, SparseMultilinearExtension};
use ark_poly_commit::LabeledCommitment;
use rand_chacha::ChaCha20Rng;

use crate::PCSNaysayer;

pub(crate) fn rand_point<F: Field>(num_vars: Option<usize>, rng: &mut ChaCha20Rng) -> Vec<F> {
    match num_vars {
        Some(n) => (0..n).map(|_| F::rand(rng)).collect(),
        None => unimplemented!(), // should not happen!
    }
}

pub(crate) fn rand_poly<Fr: PrimeField>(
    _: usize,
    num_vars: Option<usize>,
    rng: &mut ChaCha20Rng,
) -> SparseMultilinearExtension<Fr> {
    match num_vars {
        Some(n) => SparseMultilinearExtension::rand(n, rng),
        None => unimplemented!(), // should not happen in ML case!
    }
}

// Convenience function that generates a possibly dishonest LinearCodePCS
// proof based on the `dishonesty` argument, checks that
// LinearCodePCS::check outputs the expected result (typically rejection)
// and also verifies that naysay detects the planted dishonesty and
// naysay_verify accepts the naysayer proof
pub(crate) fn test_naysay_aux<'a, F, P, PCSN>(
    vk: &PCSN::VerifierKey,
    coms: impl IntoIterator<Item = &'a LabeledCommitment<PCSN::Commitment>> + Clone,
    point: &'a P::Point,
    values: impl IntoIterator<Item = F> + Clone,
    sponge: &mut impl CryptographicSponge,
    proof: PCSN::Proof,
    expected_naysayer_proof: Option<PCSN::NaysayerProof>,
) where
    F: PrimeField,
    P: Polynomial<F>,
    PCSN: PCSNaysayer<F, P>,
    PCSN::Commitment: 'a,
{
    // TODO This cumbersome block is due to the current inconsistent
    // behaviour of LinearCodePCS::check, which can return Err when
    // there is a genuine runtime error during verification OR when no
    // runtime errors occur but the proof is rejected; and, in the
    // latter case, sometimes Ok(false) is returned instead. The block
    // below can be made cleaner once open is made consistent.
    let result = PCSN::check(
        vk,
        coms.clone(),
        point,
        values.clone(),
        &proof,
        &mut sponge.clone(),
        None,
    );

    if expected_naysayer_proof.is_none() {
        assert!(result.unwrap());
    } else {
        assert!(result.is_err() || !result.unwrap());
    }

    // Produce a naysayer proof from the given PCS proof
    let naysayer_proof = PCSN::naysay(
        vk,
        coms.clone(),
        point,
        values.clone(),
        &proof,
        &mut sponge.clone(),
        None,
    )
    .unwrap();

    assert_eq!(naysayer_proof, expected_naysayer_proof);

    // Verify the naysayer proof
    if let Some(inner_naysayer_proof) = naysayer_proof {
        assert!(PCSN::verify_naysay(
            vk,
            coms,
            point,
            values,
            &proof,
            &inner_naysayer_proof,
            &mut sponge.clone(),
            None
        )
        .unwrap());
    }
}
