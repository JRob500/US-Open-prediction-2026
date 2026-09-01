from pydantic import BaseModel, Field, field_validator

VALID_ROUNDS = {"R128", "R64", "R32", "R16", "QF", "SF", "F"}
 
class PredictRequest(BaseModel):
    tour: str = Field(..., description="'atp' or 'wta'")
    player_a: str = Field(..., description="Player A's name, must match /players list")
    player_b: str = Field(..., description="Player B's name, must match /players list")
    round: str = Field(default="R128", description="Tournament round, e.g. R128, R64, ... F")
 

@field_validator("round")
@classmethod
def normalize_and_validate_round(cls, value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in VALID_ROUNDS:
        raise ValueError(
            f"'{value}' is not a valid round. Must be one of: {sorted(VALID_ROUNDS)}"
        )
    return normalized

@field_validator("tour")
@classmethod
def normalize_tour(cls, value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")
    return normalized

class PredictResponse(BaseModel):
    tour: str
    player_a: str
    player_b: str
    player_a_win_probability: float
    player_b_win_probability: float
    player_a_elo: float
    player_b_elo: float
    player_a_rank: float | None
    player_b_rank: float | None
 
 
class PlayersResponse(BaseModel):
    tour: str
    players: list[str]