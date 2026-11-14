"""RunPod calls for network volumes and for finding a data center with a GPU in stock, used by the overnight chain launcher and the retrieval
script. Pods themselves are created, read and deleted by scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py's
RunPodRestClient, which the classes here extend.

Endpoints (checked against docs.runpod.io on 2026-11-13; never called while this was written, only against fakes):
  POST   https://rest.runpod.io/v1/networkvolumes          body {name, size (GB, integer 0 to 4000), dataCenterId}; returns {id, name, size, dataCenterId}
  GET    https://rest.runpod.io/v1/networkvolumes[/<id>]   list or get
  DELETE https://rest.runpod.io/v1/networkvolumes/<id>     204 on success (never called by this project's code: the delete command is only printed)
  GET    https://api.runpod.io/v2/catalog/datacenters?include=GPU_AVAILABILITY   returns {"dataCenters": [{id, name, networkVolumeTypes,
         gpuAvailability: [{id, name, availability: NONE|LOW|MEDIUM|HIGH}]}]}
         UNVERIFIED: the documentation names this v2 catalog endpoint and bearer authentication, but it was not called, so the host, the exact
         gpuAvailability ids (assumed "NVIDIA H200") and whether the same API key works for v2 are unconfirmed. If the call fails the launcher
         says so and asks for --data-center-id.
  POST   https://rest.runpod.io/v1/pods                    adds networkVolumeId, dataCenterIds, dataCenterPriority ("custom"), volumeMountPath; for CPU pods
         computeType "CPU" with cpuFlavorIds and vcpuCount.
Storage price (docs.runpod.io/pods/pricing): network volume 0.07 dollars per GB per month under 1 TB; container disk 0.10 dollars per GB per month while running.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.log_probability_persona_check.launch_and_stop_runpod_scoring_pod import RUNPOD_REST_BASE_URL, RunPodRestClient  # noqa: E402

RUNPOD_CATALOG_BASE_URL = "https://api.runpod.io/v2/catalog"
NETWORK_VOLUME_DOLLARS_PER_GIGABYTE_MONTH = 0.07
AVAILABILITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


class RunPodClientWithNetworkVolumes(RunPodRestClient):
    def list_data_centers_with_gpu_availability(self):
        response = self.requests.get(f"{RUNPOD_CATALOG_BASE_URL}/datacenters", headers=self.headers,
                                     params={"include": "GPU_AVAILABILITY"}, timeout=60)
        if response.status_code >= 300:
            raise RuntimeError(f"RunPod data center listing failed ({response.status_code}): {response.text[:500]}")
        return response.json()["dataCenters"]

    def create_network_volume(self, name, size_gigabytes, data_center_id):
        response = self.requests.post(f"{RUNPOD_REST_BASE_URL}/networkvolumes", headers=self.headers, timeout=60,
                                      json=build_network_volume_creation_request(name, size_gigabytes, data_center_id))
        if response.status_code >= 300:
            raise RuntimeError(f"RunPod network volume creation failed ({response.status_code}): {response.text[:500]}")
        return response.json()

    def get_network_volume(self, network_volume_id):
        response = self.requests.get(f"{RUNPOD_REST_BASE_URL}/networkvolumes/{network_volume_id}", headers=self.headers, timeout=60)
        if response.status_code >= 300:
            raise RuntimeError(f"RunPod get network volume failed ({response.status_code}): {response.text[:500]}")
        return response.json()


def build_network_volume_creation_request(name, size_gigabytes, data_center_id):
    return {"name": name[:191], "size": int(size_gigabytes), "dataCenterId": data_center_id}


def delete_network_volume_command_text(network_volume_id):
    """Text only. The key is read from the project .env by whoever runs it; nothing in this project runs this."""
    return (f"curl -s -X DELETE -H \"Authorization: Bearer $RUNPOD_API_KEY\" {RUNPOD_REST_BASE_URL}/networkvolumes/{network_volume_id}"
            "   # with RUNPOD_API_KEY from the project .env; irreversible, deletes everything on the volume")


def monthly_storage_cost_dollars(size_gigabytes):
    return size_gigabytes * NETWORK_VOLUME_DOLLARS_PER_GIGABYTE_MONTH


def data_centers_with_gpu_in_stock(data_centers, gpu_name_fragment="H200", minimum_level="LOW"):
    """Data center ids that list a GPU whose id or name contains the fragment at the minimum availability or better, most available first
    (ties keep the listing order)."""
    candidates = []
    for position, data_center in enumerate(data_centers):
        best = 0
        for gpu in data_center.get("gpuAvailability") or []:
            if gpu_name_fragment.lower() in f"{gpu.get('id', '')} {gpu.get('name', '')}".lower():
                best = max(best, AVAILABILITY_RANK.get(gpu.get("availability"), 0))
        supports_network_volumes = bool(data_center.get("networkVolumeTypes"))
        if best >= AVAILABILITY_RANK[minimum_level] and supports_network_volumes:
            candidates.append((-best, position, data_center["id"]))
    return [data_center_id for _, _, data_center_id in sorted(candidates)]
