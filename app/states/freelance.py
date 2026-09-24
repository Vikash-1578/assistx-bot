"""FSM states for the freelance proposal workflow."""
from aiogram.fsm.state import State, StatesGroup


class FreelanceProposal(StatesGroup):
    waiting_client_name = State()
    waiting_project_desc = State()
    waiting_budget = State()
    waiting_deadline = State()
    waiting_tone = State()
