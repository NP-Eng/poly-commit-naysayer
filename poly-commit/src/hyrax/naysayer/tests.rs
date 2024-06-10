use crate::hyrax::{
    naysayer::HyraxNaysayerProof, utils::tensor_prime, Error, HyraxCommitment,
    HyraxCommitmentState, HyraxCommitterKey, HyraxPC, HyraxProof, HyraxVerifierKey,
    LabeledCommitment, PolynomialCommitment,
};
use crate::utils::{inner_product, scalar_by_vector, test_sponge, vector_sum};
use crate::LabeledPolynomial;
use ark_bls12_377::G1Affine;
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::serialize_to_vec;
use ark_std::test_rng;
use ark_std::{rand::RngCore, vec::Vec, UniformRand};
use blake2::Blake2s256;
use digest::Digest;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// PCS definition
type Fq = <G1Affine as AffineRepr>::ScalarField;
type Hyrax377 = HyraxPC<G1Affine, DenseMultilinearExtension<Fq>>;

// Types of dishonesty that can be introduce in a LinearCodePCS proof
#[derive(Eq, PartialEq)]
enum HyraxDishonesty {
    // No dishonesty: same as the open
    None,
    // Modify the commitment beta to <r, d>
    DishonestBetaCom,
}

impl<G, P> HyraxPC<G, P>
where
    G: AffineRepr + Absorb,
    G::ScalarField: Absorb,
    P: MultilinearExtension<G::ScalarField>,
{
    // Convenience function that generates a possibly dishonest LinearCodePCS
    // proof based on the `dishonesty` argument, checks that
    // LinearCodePCS::check outputs the expected result (typically rejection)
    // and also verifies that naysay detects the planted dishonesty and
    // naysay_verify accepts the naysayer proof
    fn test_naysay_aux(
        ck: &HyraxCommitterKey<G>,
        vk: &HyraxVerifierKey<G>,
        l_com: &LabeledCommitment<HyraxCommitment<G>>,
        com_state: &HyraxCommitmentState<G::ScalarField>,
        point: &P::Point,
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
        dishonesty: HyraxDishonesty,
        expected_naysayer_proof: HyraxNaysayerProof,
    ) {
        // Generating possibly dishonest proof
        let proof = Self::dishonest_open(
            ck,
            l_com,
            com_state,
            point,
            &mut sponge.clone(),
            rng,
            dishonesty,
        )
        .unwrap();

        let v_proof = vec![proof];

        // TODO This cumbersome block is due to the current inconsistent
        // behaviour of LinearCodePCS::check, which can return Err when
        // there is a genuine runtime error during verification OR when no
        // runtime errors occur but the proof is rejected; and, in the
        // latter case, sometimes Ok(false) is returned instead. The block
        // below can be made cleaner once open is made consistent.
        let result = Self::check(vk, [l_com], point, [], &v_proof, &mut sponge.clone(), None);

        assert_eq!(
            expected_naysayer_proof == HyraxNaysayerProof::Aye,
            result.unwrap()
        );

        // Produce a naysayer proof from the given PCS proof
        let naysayer_proof =
            Self::naysay(vk, [l_com], point, [], &v_proof, &mut sponge.clone(), None).unwrap();

        assert_eq!(naysayer_proof, expected_naysayer_proof);

        // Verify the naysayer proof
        assert_eq!(
            expected_naysayer_proof == HyraxNaysayerProof::Aye,
            !Self::naysayer_verify(
                vk,
                [l_com],
                point,
                [],
                &v_proof,
                naysayer_proof,
                &mut sponge.clone(),
                None
            )
            .unwrap()
        );
    }

    fn dishonest_open(
        ck: &HyraxCommitterKey<G>,
        l_com: &LabeledCommitment<HyraxCommitment<G>>,
        com_state: &HyraxCommitmentState<G::ScalarField>,
        point: &P::Point,
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
        dishonesty: HyraxDishonesty,
    ) -> Result<HyraxProof<G>, Error> {
        let n = point.len();

        if n % 2 == 1 {
            // Only polynomials with an even number of variables are
            // supported in this implementation
            return Err(Error::InvalidNumberOfVariables);
        }

        let dim = 1 << n / 2;

        // Reversing the point is necessary because the MLE interface returns
        // evaluations in little-endian order
        let point_rev: Vec<G::ScalarField> = point.iter().rev().cloned().collect();

        let point_lower = &point_rev[n / 2..];
        let point_upper = &point_rev[..n / 2];

        // Deriving the tensors which result in the evaluation of the polynomial
        // when they are multiplied by the coefficient matrix.
        let l = tensor_prime(point_lower);
        let r = tensor_prime(point_upper);

        let rng_inner = rng.expect("Opening polynomials requires randomness");

        let com = l_com.commitment();

        // Absorbing public parameters
        sponge.absorb(
            &Blake2s256::digest(serialize_to_vec!(*ck).map_err(|_| Error::TranscriptError)?)
                .as_slice(),
        );

        // Absorbing the commitment to the polynomial
        sponge.absorb(&serialize_to_vec!(com.row_coms).map_err(|_| Error::TranscriptError)?);

        // Absorbing the point
        sponge.absorb(point);

        // Commiting to the matrix formed by the polynomial coefficients
        let t = &com_state.mat;

        let lt = t.row_mul(&l);

        // t_prime coincides witht he Pedersen commitment to lt with the
        // randomnes r_lt computed here
        let r_lt = cfg_iter!(l)
            .zip(&com_state.randomness)
            .map(|(l, r)| *l * r)
            .sum::<G::ScalarField>();

        let eval = inner_product(&lt, &r);

        // Singleton commit
        let (com_eval, r_eval) = {
            let r = G::ScalarField::rand(rng_inner);
            ((ck.com_key[0] * eval + ck.h * r).into(), r)
        };

        // ******** Dot product argument ********
        // Appendix A.2 in the reference article

        let d: Vec<G::ScalarField> = (0..dim).map(|_| G::ScalarField::rand(rng_inner)).collect();

        let b = inner_product(&r, &d);

        // Multi-commit
        let r_d = G::ScalarField::rand(rng_inner);
        let com_d = (Self::pedersen_commit(&ck.com_key, &d) + ck.h * r_d).into();

        // Singleton commit
        let r_b = G::ScalarField::rand(rng_inner);
        let mut com_b = (ck.com_key[0] * b + ck.h * r_b).into();

        if dishonesty == HyraxDishonesty::DishonestBetaCom {
            com_b = (com_b + ck.com_key[0]).into();
        }

        // Absorbing the commitment to the evaluation
        sponge.absorb(&serialize_to_vec!(com_eval).map_err(|_| Error::TranscriptError)?);

        // Absorbing the two auxiliary commitments
        sponge.absorb(&serialize_to_vec!(com_d).map_err(|_| Error::TranscriptError)?);
        sponge.absorb(&serialize_to_vec!(com_b).map_err(|_| Error::TranscriptError)?);

        // Receive the random challenge c from the verifier, i.e. squeeze
        // it from the transcript.
        let c = sponge.squeeze_field_elements(1)[0];

        let z = vector_sum(&d, &scalar_by_vector(c, &lt));
        let z_d = c * r_lt + r_d;
        let z_b = c * r_eval + r_b;

        Ok(HyraxProof {
            com_eval,
            com_d,
            com_b,
            z,
            z_d,
            z_b,
        })
    }
}

#[test]
fn test_naysay() {
    // Desired number of variables (must be even!)
    let n = 8;

    let chacha = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();

    let pp = Hyrax377::setup(1, Some(n), chacha).unwrap();

    let (ck, vk) = Hyrax377::trim(&pp, 1, 1, None).unwrap();

    let l_poly = LabeledPolynomial::new(
        "test_poly".to_string(),
        DenseMultilinearExtension::rand(n, chacha),
        None,
        None,
    );

    let (c, rands) = Hyrax377::commit(&ck, &[l_poly.clone()], Some(chacha)).unwrap();

    let point: Vec<Fq> = rand_point(n, chacha);

    // Dummy argument
    let test_sponge = test_sponge::<Fq>();

    // The only arguments to test_naysay_aux we intend to change are the
    // dishonesty and the expected type of naysayer proof returned
    let mut test_naysay_with = |dishonesty, expected_naysayer_proof| {
        Hyrax377::test_naysay_aux(
            &ck,
            &vk,
            &c[0],
            &rands[0],
            &point,
            &mut test_sponge.clone(),
            Some(chacha),
            dishonesty,
            expected_naysayer_proof,
        );
    };

    /***************** Case 1 *****************/
    // Honest proof verifies and is not naysaid
    test_naysay_with(HyraxDishonesty::None, HyraxNaysayerProof::Aye);

    /***************** Case 2 *****************/
    // Modify the commitment to beta = <r, d>
    test_naysay_with(
        HyraxDishonesty::DishonestBetaCom,
        HyraxNaysayerProof::SecondCheckLie,
    );
}

fn rand_point<F: PrimeField>(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<F> {
    (0..num_vars).map(|_| F::rand(rng)).collect()
}
