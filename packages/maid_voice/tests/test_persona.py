from pathlib import Path

from maid_voice.persona import Expression, RoleCard

CARD = Path(__file__).resolve().parents[2] / "maid_agent" / "persona" / "natsume.v2.toml"


def test_role_card_builds_replyer_prompt_without_planner_rules():
    card = RoleCard.load(CARD)
    prompt = card.replyer_prompt([Expression("相手が謝ってきたとき", "「別に」で軽く流す", "別に。気にしないで。")])
    assert card.user_name == "あきら" and "ワタシ" in prompt and "あきら" in prompt
    assert "相手が謝ってきたとき" in prompt and "別に。気にしないで。" in prompt
    assert card.behavior and card.behavior not in prompt  # behaviour rules are for the planner only
