# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.interfaces import is_mixture_of_experts


class _FakeMoELanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expert_weights: list[list[torch.Tensor]] = []
        self.moe_layers = [nn.Identity()]
        self.num_moe_layers = 1
        self.num_expert_groups = 1
        self.num_logical_experts = 4
        self.num_physical_experts = 4
        self.num_local_physical_experts = 2
        self.num_routed_experts = 4
        self.num_shared_experts = 0
        self.num_redundant_experts = 0
        self.eplb_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state = (
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )
        self.expert_weights = [[torch.empty(1)]]

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts


def test_gemma4_conditional_generation_delegates_moe_interface() -> None:
    language_model = _FakeMoELanguageModel()
    model = Gemma4ForConditionalGeneration.__new__(Gemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = language_model

    for attr in (
        "expert_weights",
        "moe_layers",
        "num_moe_layers",
        "num_expert_groups",
        "num_logical_experts",
        "num_physical_experts",
        "num_local_physical_experts",
        "num_routed_experts",
        "num_shared_experts",
        "num_redundant_experts",
    ):
        setattr(model, attr, getattr(language_model, attr))

    assert is_mixture_of_experts(model)

    expert_load_view = torch.zeros((1, 4), dtype=torch.int32)
    logical_to_physical_map = torch.arange(4).reshape(1, 4, 1)
    logical_replica_count = torch.ones((1, 4), dtype=torch.long)
    model.set_eplb_state(
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
    )

    assert language_model.eplb_state == (
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
    )
    assert model.expert_weights is language_model.expert_weights

    model.update_physical_experts_metadata(6, 3)
    assert model.num_physical_experts == language_model.num_physical_experts == 6
    assert (
        model.num_local_physical_experts
        == language_model.num_local_physical_experts
        == 3
    )
    assert model.num_redundant_experts == language_model.num_redundant_experts == 2
