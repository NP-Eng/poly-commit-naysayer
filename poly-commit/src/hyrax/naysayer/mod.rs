use crate::hyrax::{
    utils::tensor_prime, Error, HyraxCommitment, HyraxPC, HyraxProof, HyraxVerifierKey,
    LabeledCommitment,
};
use crate::utils::inner_product;
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::PrimeField;
use ark_poly::MultilinearExtension;
use ark_serialize::serialize_to_vec;
use ark_std::{rand::RngCore, vec::Vec};
use blake2::Blake2s256;
use digest::Digest;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(test)]
mod tests;

/// Naysayer proof for Hyrax opening, indicating which piece of the opening
/// proof is incorrect (if any)
#[derive(PartialEq, Debug)]
pub enum HyraxNaysayerProof {
    /// No errors, opening proof accepted
    Aye,
    /// The first check in the dot-product argument ([Hyrax] figure 6, equation
    /// (13)) fails
    FirstCheckLie,
    /// The second check in the dot-product argument ([Hyrax] figure 6, equation
    /// (14)) fails
    SecondCheckLie,
}

impl<G, P> HyraxPC<G, P>
where
    G: AffineRepr + Absorb,
    G::ScalarField: Absorb,
    P: MultilinearExtension<G::ScalarField>,
{
    /// Indicate whether the given proofs are valid or point to their issues,
    /// producing a naysayer proof
    pub fn naysay<'a>(
        vk: &HyraxVerifierKey<G>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<HyraxCommitment<G>>>,
        point: &'a P::Point,
        _values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Vec<HyraxProof<G>>,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<HyraxNaysayerProof, Error> {
        let n = point.len();

        if n % 2 == 1 {
            // Only polynomials with an even number of variables are
            // supported in this implementation
            return Err(Error::InvalidNumberOfVariables);
        }

        // Reversing the point is necessary because the MLE interface returns
        // evaluations in little-endian order
        let point_rev: Vec<G::ScalarField> = point.iter().rev().cloned().collect();

        let point_lower = &point_rev[n / 2..];
        let point_upper = &point_rev[..n / 2];

        // Deriving the tensors which result in the evaluation of the polynomial
        // when they are multiplied by the coefficient matrix.
        let l = tensor_prime(point_lower);
        let r = tensor_prime(point_upper);

        for (com, h_proof) in commitments.into_iter().zip(proof.iter()) {
            let row_coms = &com.commitment().row_coms;

            // extract each field from h_proof
            let HyraxProof {
                com_eval,
                com_d,
                com_b,
                z,
                z_d,
                z_b,
            } = h_proof;

            if row_coms.len() != 1 << n / 2 {
                return Err(Error::IncorrectCommitmentSize {
                    encountered: row_coms.len(),
                    expected: 1 << n / 2,
                });
            }

            // Absorbing public parameters
            sponge.absorb(
                &Blake2s256::digest(serialize_to_vec!(*vk).map_err(|_| Error::TranscriptError)?)
                    .as_slice(),
            );

            // Absorbing the commitment to the polynomial
            sponge.absorb(&serialize_to_vec!(*row_coms).map_err(|_| Error::TranscriptError)?);

            // Absorbing the point
            sponge.absorb(point);

            // Absorbing the commitment to the evaluation
            sponge.absorb(&serialize_to_vec!(*com_eval).map_err(|_| Error::TranscriptError)?);

            // Absorbing the two auxiliary commitments
            sponge.absorb(&serialize_to_vec!(*com_d).map_err(|_| Error::TranscriptError)?);
            sponge.absorb(&serialize_to_vec!(*com_b).map_err(|_| Error::TranscriptError)?);

            // Receive the random challenge c from the verifier, i.e. squeeze
            // it from the transcript.
            let c: G::ScalarField = sponge.squeeze_field_elements(1)[0];

            // Second check from the paper (figure 6, equation (14))
            // Moved here for potential early return
            let com_dp = (vk.com_key[0] * inner_product(&r, z) + vk.h * z_b).into();
            if com_dp != (com_eval.mul(c) + com_b).into() {
                return Ok(HyraxNaysayerProof::SecondCheckLie);
            }

            // Computing t_prime with a multi-exponentiation
            let l_bigint = cfg_iter!(l)
                .map(|chi| chi.into_bigint())
                .collect::<Vec<_>>();
            let t_prime: G = <G::Group as VariableBaseMSM>::msm_bigint(&row_coms, &l_bigint).into();

            // First check from the paper (figure 6, equation (13))
            let com_z_zd = (Self::pedersen_commit(&vk.com_key, z) + vk.h * z_d).into();
            if com_z_zd != (t_prime.mul(c) + com_d).into() {
                return Ok(HyraxNaysayerProof::SecondCheckLie);
            }
        }

        Ok(HyraxNaysayerProof::Aye)
    }

    /// Verifies the naysayer proof.
    /// - Returns `Ok(true)` if the original proof is rejected (i.e. the
    ///   naysayer proof points to a valid issue).
    /// - Returns `Ok(false)` if the original proof is not rejected, i.e.
    ///     - either the naysayer proof told to accept the original proof
    ///       ("Aye")
    ///     - or the naysayer proof points to an invalid issue
    /// - Returns `Err` if another type of error occurs during verification of
    ///   the naysayer proof.
    pub fn naysayer_verify<'a>(
        vk: &HyraxVerifierKey<G>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<HyraxCommitment<G>>>,
        point: &'a P::Point,
        _values: impl IntoIterator<Item = G::ScalarField>, // Unused! V does not learn the evaluation
        proof: &Vec<HyraxProof<G>>,
        naysayer_proof: HyraxNaysayerProof,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Error> {
        let commitments: Vec<&LabeledCommitment<HyraxCommitment<G>>> =
            commitments.into_iter().collect();
        assert!(commitments.len() == proof.len());
        assert!(commitments.len() == 1);

        let commitment = commitments[0].commitment();
        let row_coms = &commitment.row_coms;

        // extract each field from the original proof
        let HyraxProof {
            com_eval,
            com_d,
            com_b,
            z,
            z_d,
            z_b,
        } = &proof[0];

        if HyraxNaysayerProof::Aye == naysayer_proof {
            return Ok(false);
        }

        let n = point.len();

        // Reversing the point is necessary because the MLE interface returns
        // evaluations in little-endian order
        let point_rev: Vec<G::ScalarField> = point.iter().rev().cloned().collect();

        // Absorbing public parameters
        sponge.absorb(
            &Blake2s256::digest(serialize_to_vec!(*vk).map_err(|_| Error::TranscriptError)?)
                .as_slice(),
        );

        // Absorbing the commitment to the polynomial
        sponge.absorb(&serialize_to_vec!(*row_coms).map_err(|_| Error::TranscriptError)?);

        // Absorbing the point
        sponge.absorb(point);

        // Absorbing the commitment to the evaluation
        sponge.absorb(&serialize_to_vec!(*com_eval).map_err(|_| Error::TranscriptError)?);

        // Absorbing the two auxiliary commitments
        sponge.absorb(&serialize_to_vec!(*com_d).map_err(|_| Error::TranscriptError)?);
        sponge.absorb(&serialize_to_vec!(*com_b).map_err(|_| Error::TranscriptError)?);

        // Receive the random challenge c from the verifier, i.e. squeeze
        // it from the transcript.
        let c: G::ScalarField = sponge.squeeze_field_elements(1)[0];

        match naysayer_proof {
            HyraxNaysayerProof::Aye => Ok(false),
            HyraxNaysayerProof::FirstCheckLie => {
                // Take reversal into account
                let point_lower = &point_rev[n / 2..];
                let l = tensor_prime(point_lower);

                let l_bigint = cfg_iter!(l)
                    .map(|chi| chi.into_bigint())
                    .collect::<Vec<_>>();
                let t_prime: G =
                    <G::Group as VariableBaseMSM>::msm_bigint(&row_coms, &l_bigint).into();

                // First check from the paper (figure 6, equation (13))
                let com_z_zd = (Self::pedersen_commit(&vk.com_key, z) + vk.h * z_d).into();
                Ok(com_z_zd != (t_prime.mul(c) + com_d).into())
            }
            HyraxNaysayerProof::SecondCheckLie => {
                // Take reversal into account
                let point_upper = &point_rev[..n / 2];
                let r = tensor_prime(point_upper);

                // Second check from the paper (figure 6, equation (14))
                let com_dp = (vk.com_key[0] * inner_product(&r, z) + vk.h * z_b).into();
                Ok(com_dp != (com_eval.mul(c) + com_b).into())
            }
        }
    }
}
