use alloc::vec::Vec;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::{Field, PrimeField};
use ark_poly::{MultilinearExtension, Polynomial, SparseMultilinearExtension};
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial};
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

pub(crate) fn test_invalid_naysayer_proofs<'a, F, P, PCSN>(
    vk: &PCSN::VerifierKey,
    ck: &PCSN::CommitterKey,
    labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>> + Clone,
    coms: impl IntoIterator<Item = &'a LabeledCommitment<PCSN::Commitment>> + Clone,
    com_states: impl IntoIterator<Item = &'a PCSN::CommitmentState> + Clone,
    point: &'a P::Point,
    sponge: &mut impl CryptographicSponge,
    naysayer_proofs: Vec<PCSN::NaysayerProof>,
) where
    F: PrimeField,
    P: Polynomial<F> + 'a,
    PCSN: PCSNaysayer<F, P>,
    PCSN::Commitment: 'a,
    PCSN::CommitmentState: 'a,
{
    let values = labeled_polynomials
        .clone()
        .into_iter()
        .map(|lp| lp.evaluate(point))
        .collect::<Vec<_>>();

    let valid_proof = PCSN::open(
        ck,
        labeled_polynomials,
        coms.clone(),
        point,
        &mut sponge.clone(),
        com_states,
        None,
    )
    .unwrap();

    let assert_invalid_naysayer_proof = |naysayer_proof| {
        assert!(!PCSN::verify_naysay(
            vk,
            coms.clone(),
            point,
            values.clone(),
            &valid_proof,
            &naysayer_proof,
            &mut sponge.clone(),
            None
        )
        .unwrap());
    };

    for naysayer_proof in naysayer_proofs {
        assert_invalid_naysayer_proof(naysayer_proof);
    }
}
