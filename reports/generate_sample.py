"""Generate sample outputs for the reports/ directory.

Sample 1: the five missing-value scenarios on a local repo.
Sample 2: a partial batch failure (one entity's read batch keeps failing)
showing the other rows still returned and the failed batch listed.
"""

import json
import sys

sys.path.insert(0, "sdk/python")
sys.path.insert(0, "sdk/python/tests")

from feast.infra.online_stores.sqlite import SqliteOnlineStore
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import OnlineStatusConfig
from feast.utils import _utc_now
from tests.utils.cli_repo_creator import CliRunner, get_example_repo


def _write(store, fv, driver_id, features):
    store._get_provider().online_write_batch(
        config=store.config,
        table=fv,
        data=[
            (
                EntityKeyProto(
                    join_keys=["driver_id"],
                    entity_values=[ValueProto(int64_val=driver_id)],
                ),
                features,
                _utc_now(),
                _utc_now(),
            )
        ],
        progress=None,
    )


def main() -> None:
    runner = CliRunner()
    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"), "file"
    ) as store:
        fv = store.get_feature_view(name="driver_locations")
        _write(store, fv, 1, {"lat": ValueProto(float_val=0.5)})
        _write(store, fv, 2, {"lat": ValueProto(), "lon": ValueProto(string_val="2.0")})
        _write(store, fv, 3, {"lat": ValueProto(string_val="oops")})
        # driver 4: nothing written -> entity not in store

        store.config.online_status_config = OnlineStatusConfig(enabled=True)
        response = store.get_online_features(
            features=[
                "driver_locations:lat",
                "driver_locations:lon",
                "ghost_view:foo",
            ],
            entity_rows=[{"driver_id": i} for i in (1, 2, 3, 4)],
        )
        sample1 = {
            "values": response.to_dict(),
            "status_report": response.status_report.to_dict(),
        }

        # --- sample 2: partial batch failure ---
        store.config.online_status_config = OnlineStatusConfig(
            enabled=True, read_batch_size=1, max_batch_retries=1
        )
        original = SqliteOnlineStore.online_read

        def flaky(self, config, table, entity_keys, requested_features=None):
            for ek in entity_keys:
                for v in ek.entity_values:
                    if v.int64_val == 2:
                        raise RuntimeError("simulated store failure")
            return original(self, config, table, entity_keys, requested_features)

        SqliteOnlineStore.online_read = flaky
        try:
            response2 = store.get_online_features(
                features=["driver_locations:lat"],
                entity_rows=[{"driver_id": 1}, {"driver_id": 2}, {"driver_id": 3}],
            )
        finally:
            SqliteOnlineStore.online_read = original
        sample2 = {
            "values": response2.to_dict(),
            "status_report": response2.status_report.to_dict(),
        }

        print(json.dumps({"sample1_five_scenarios": sample1, "sample2_partial_failure": sample2}, indent=2, default=str))


if __name__ == "__main__":
    main()
