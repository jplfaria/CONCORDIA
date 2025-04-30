import pytest
from concord.llm.argo_gateway import ArgoGatewayClient, llm_label
from concord.llm.prompts import LABEL_SET

@pytest.mark.integration
def test_llm_roundtrip():
    client = ArgoGatewayClient(model="gpto3mini", retries=0)  # quick ping
    label = llm_label("DNA ligase", "NAD-dependent DNA ligase", client)
    assert label in LABEL_SET
