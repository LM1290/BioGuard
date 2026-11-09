import unittest
from itertools import combinations

import pandas as pd

from bioguard.data_loader import (
    ACTIVE_SPLITS,
    DataContractError,
    PUSamplingConfig,
    SplitConfig,
    _scaffold_key,
    assign_relation_splits,
    build_binary_pu_dataset,
    prepare_twosides_relations,
)


SCAFFOLD_GROUPS = [
    ["Oc1ccccc1", "Clc1ccccc1", "Cc1ccccc1"],
    ["Oc1ccncc1", "Clc1ccncc1", "Cc1ccncc1"],
    ["OC1CCCCC1", "ClC1CCCCC1", "CC1CCCCC1"],
    ["c1ccoc1", "Cc1ccoc1", "Clc1ccoc1"],
    ["c1ccsc1", "Cc1ccsc1", "Clc1ccsc1"],
    ["c1ncc[nH]1", "Cc1ncc[nH]1", "Clc1ncc[nH]1"],
]


def raw_row(drug_a, smiles_a, event, drug_b, smiles_b):
    return {
        "Drug1_ID": drug_a,
        "Drug1": smiles_a,
        "Y": event,
        "Drug2_ID": drug_b,
        "Drug2": smiles_b,
    }


def sparse_scaffold_fixture():
    rows = []
    ids_by_group = []
    for group_index, smiles_group in enumerate(SCAFFOLD_GROUPS):
        ids = [f"D{group_index}_{index}" for index in range(3)]
        ids_by_group.append(ids)
        rows.append(
            raw_row(
                ids[0],
                smiles_group[0],
                100 + group_index,
                ids[1],
                smiles_group[1],
            )
        )
    for group_index in range(len(SCAFFOLD_GROUPS)):
        next_group = (group_index + 1) % len(SCAFFOLD_GROUPS)
        rows.append(
            raw_row(
                ids_by_group[group_index][2],
                SCAFFOLD_GROUPS[group_index][2],
                900 + group_index,
                ids_by_group[next_group][2],
                SCAFFOLD_GROUPS[next_group][2],
            )
        )
    return pd.DataFrame(rows)


def dense_fixture():
    drugs = [
        (f"D{group_index}_{drug_index}", smiles)
        for group_index, group in enumerate(SCAFFOLD_GROUPS)
        for drug_index, smiles in enumerate(group)
    ]
    return pd.DataFrame(
        [
            raw_row(
                drug_a,
                smiles_a,
                100 + pair_index % 5,
                drug_b,
                smiles_b,
            )
            for pair_index, (
                (drug_a, smiles_a),
                (drug_b, smiles_b),
            ) in enumerate(combinations(drugs, 2))
        ]
    )


class RelationContractTests(unittest.TestCase):
    def test_event_identity_is_preserved_and_exact_duplicates_are_counted(self):
        raw = pd.DataFrame(
            [
                raw_row("B", "CCN", 10, "A", "CCO"),
                raw_row("A", "CCO", 10.0, "B", "CCN"),
                raw_row("A", "CCO", 11, "B", "CCN"),
            ]
        )
        bundle = prepare_twosides_relations(raw)

        self.assertEqual(len(bundle.records), 2)
        self.assertEqual(set(bundle.records["event_id"]), {"10", "11"})
        counts = dict(
            zip(
                bundle.records["event_id"],
                bundle.records["source_row_count"],
            )
        )
        self.assertEqual(counts, {"10": 2, "11": 1})
        self.assertEqual(bundle.records["pair_id"].nunique(), 1)
        self.assertNotIn("label", bundle.records.columns)

    def test_missing_event_is_rejected(self):
        raw = pd.DataFrame(
            [raw_row("A", "CCO", None, "B", "CCN")]
        )
        with self.assertRaisesRegex(DataContractError, "missing event"):
            prepare_twosides_relations(raw)

    def test_scaffold_split_has_no_active_drug_or_scaffold_overlap(self):
        relations = prepare_twosides_relations(
            sparse_scaffold_fixture()
        ).records
        bundle = assign_relation_splits(
            relations,
            SplitConfig(method="scaffold_disjoint", seed=17),
        )
        active = bundle.records[
            bundle.records["split"].isin(ACTIVE_SPLITS)
        ]

        drug_sets = {}
        scaffold_sets = {}
        for split in ACTIVE_SPLITS:
            subset = active[active["split"] == split]
            drug_sets[split] = set(subset["drug_a"]) | set(subset["drug_b"])
            scaffold_sets[split] = {
                _scaffold_key(smiles)
                for smiles in pd.concat(
                    [subset["smiles_a"], subset["smiles_b"]]
                )
            }

        for index, left in enumerate(ACTIVE_SPLITS):
            for right in ACTIVE_SPLITS[index + 1:]:
                self.assertFalse(drug_sets[left] & drug_sets[right])
                self.assertFalse(scaffold_sets[left] & scaffold_sets[right])
        self.assertGreater(
            bundle.manifest["excluded_cross_partition_relations"],
            0,
        )

    def test_drug_disjoint_split_has_no_active_drug_overlap(self):
        relations = prepare_twosides_relations(dense_fixture()).records
        bundle = assign_relation_splits(
            relations,
            SplitConfig(method="drug_disjoint", seed=12),
        )
        self.assertEqual(
            bundle.manifest["drug_overlap"],
            {"train_val": 0, "train_test": 0, "val_test": 0},
        )

    def test_all_events_for_a_pair_share_one_random_split(self):
        raw = dense_fixture()
        extra = raw.iloc[[0]].copy()
        extra["Y"] = 999
        relations = prepare_twosides_relations(
            pd.concat([raw, extra], ignore_index=True)
        ).records
        bundle = assign_relation_splits(
            relations,
            SplitConfig(method="pair_random", seed=8),
        )
        self.assertTrue(
            (bundle.records.groupby("pair_id")["split"].nunique() == 1).all()
        )

    def test_binary_derivative_marks_missing_pairs_unlabeled(self):
        relations = prepare_twosides_relations(
            sparse_scaffold_fixture()
        ).records
        bundle = build_binary_pu_dataset(
            relations,
            SplitConfig(method="scaffold_disjoint", seed=23),
            PUSamplingConfig(
                train_unlabeled_ratio=1.0,
                val_unlabeled_ratio=1.0,
                test_unlabeled_ratio=1.0,
                seed=23,
            ),
        )
        records = bundle.records

        self.assertEqual(
            set(records["observation"]),
            {"positive", "unlabeled"},
        )
        self.assertNotIn("label", records.columns)
        self.assertFalse(
            bundle.manifest["unlabeled_are_confirmed_negatives"]
        )
        unlabeled = records[records["observation"] == "unlabeled"]
        self.assertTrue(
            (unlabeled["negative_evidence"] == "none_missing_from_twosides").all()
        )
        positive_pairs = set(relations["pair_id"])
        self.assertFalse(positive_pairs & set(unlabeled["pair_id"]))

    def test_sampling_and_splits_are_input_order_invariant(self):
        relations = prepare_twosides_relations(
            sparse_scaffold_fixture()
        ).records
        config = SplitConfig(method="scaffold_disjoint", seed=31)
        sampling = PUSamplingConfig(seed=31)
        first = build_binary_pu_dataset(relations, config, sampling)
        second = build_binary_pu_dataset(
            relations.sample(frac=1.0, random_state=99),
            config,
            sampling,
        )

        columns = ["sample_id", "observation", "split"]
        pd.testing.assert_frame_equal(
            first.records[columns].reset_index(drop=True),
            second.records[columns].reset_index(drop=True),
        )
        self.assertEqual(
            first.manifest["records_sha256"],
            second.manifest["records_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
