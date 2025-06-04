"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""
import pickle
import torch
from cfr_oneshot_agent import CFROneShotAgent
from collections import defaultdict

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent
from MatchingPennies import MyAgent as mp

class CFRAgent(CFROneShotAgent):
    """
    This is the only class you *need* to implement. The current skeleton simply loads a single model
    that is supposed to be saved in MODEL_PATH (train.py can be used to train such a model).
    """

    _tau   = 0.3     # learning rate for trust EMA
    _floor = 0.1     # never drive trust to zero

    def _needed(self, partner_id: str) -> int:
        """Override so that respond() always sees our *true* remaining need."""
        role = self._role(partner_id)
        if role == "B":
            return self.rem_buy
        else:
            return self.rem_sell

    def init(self):
        super().init()
        self.trust = defaultdict(lambda: 0.5)
        self._daily_reset()

    def before_step(self):
        super().before_step()
        self._daily_reset()

    def _daily_reset(self):
        self.rem_buy  = self.awi.needed_supplies
        self.rem_sell = self.awi.needed_sales
        self.outstanding = defaultdict(int)   # partner -> qty proposed today

    def _quota(self, partner_id: str, role: str) -> int:
        """Allocate portion of remaining need proportional to trust."""
        partner_ids = (self.awi.my_suppliers if role=="B"
                       else self.awi.my_consumers)
        weights = [max(self.trust[p], self._floor) for p in partner_ids]
        total_w = sum(weights)
        if total_w == 0:                      # shouldn’t happen
            total_w = len(partner_ids)*self._floor
        my_w   = max(self.trust[partner_id], self._floor)
        need   = self.rem_buy if role=="B" else self.rem_sell
        return max(1, int(round(need * my_w / total_w)))

    def propose(self, negotiator_id, state):
        role   = self._role(negotiator_id)
        print(self.outstanding)
        need   = self._needed(negotiator_id) - self.outstanding[negotiator_id]
        if need <= 0:
            return None

        offer_cap = self._quota(negotiator_id, role)
        need      = min(need, offer_cap)

        I = self._infoset(role, state, need)
        q, price = self._sample_action(I, role)
        q = min(q, need)
        self.outstanding[negotiator_id] += q
        return (q, self.awi.current_step, price)

    def respond(self, negotiator_id, state, src=""):
        decision = super().respond(negotiator_id, state, src)

        if decision == ResponseType.ACCEPT_OFFER:
            qty = state.current_offer[QUANTITY]
            role = self._role(negotiator_id)
            if role == "B":
                self.rem_buy  -= qty
            else:
                self.rem_sell -= qty
            self._update_trust(negotiator_id, accepted=True)
            self.outstanding[negotiator_id] = 0
        else:
            self._update_trust(negotiator_id, accepted=False)

        return decision

    def _update_trust(self, pid: str, accepted: bool):
        prev = self.trust[pid]
        self.trust[pid] = (1 - self._tau) * prev + self._tau * (1.0 if accepted else 0.0)
        print(f"Max trust: {max(self.trust)}", self.trust)


if __name__ == "__main__":
    import sys
    from helpers.runner import run

    run([CFRAgent,mp], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
