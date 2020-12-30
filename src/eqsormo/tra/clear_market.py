#    Copyright 2019-2020 Matthew Wigginton Conway

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Author: Matthew Wigginton Conway <matt@indicatrix.org>
#         School of Geographical Sciences and Urban Planning,
#         Arizona State University

import numpy as np

from logging import getLogger
import tempfile
import queue
import threading
import multiprocessing
import os
import pandas as pd

LOG = getLogger(__name__)


def clear_market_iter(
    non_price_utilities,
    hhidx,
    choiceidx,
    supply,
    income,
    starting_price,
    price_income_transformation,
    price_income_params,
    budget_coef,
    max_rent_to_income=None,
    convergence_criterion=1e-4,
    step=1e-2,
    weights=None,
    fixed_price=0,
    speed_control=0.25,
):
    """
    Run one iteration of the market-clearing algorithm.

    income and price should be by household index/choice index, respectively.

    returns price after next iteration, and flag for whether prices have converged. Note that once
    prices converge, one more call is necessary to confirm convergence, to avoid additional computation
    of market shares to check convergence after prices are updated.
    """
    price = starting_price

    alt_income = income[hhidx]

    if max_rent_to_income is not None:
        # could cause oscillation if price keeps going above max income
        # Also, this will affect sorting equilibrium, because the price may effectively get fixed to the
        # 90th percentile income because it always moves off, and everything else equilibrate around it
        if np.any(price > np.max(income) * max_rent_to_income):
            raise ValueError(
                "Some prices have exceeded max income - setting to be affordable to 90th"
                + "percentile household to keep process going."
            )

    shares = compute_shares(
        price=price,
        supply=supply,
        alt_income=alt_income,
        choiceidx=choiceidx,
        hhidx=hhidx,
        non_price_utilities=non_price_utilities,
        price_income_transformation=price_income_transformation,
        price_income_params=price_income_params,
        budget_coef=budget_coef,
        max_rent_to_income=max_rent_to_income,
        weights=weights,
    )

    if np.any(shares == 0):
        raise ValueError("Some shares are zero.")

    if np.allclose(shares, supply):
        return price, True

    assert np.allclose(
        np.sum(shares), np.sum(supply)
    ), "shares and supply totals do not match"

    excess_demand = shares - supply

    # maxdiff = np.max(np.abs(shares - supply) / supply)
    maxdiff = np.max(np.abs(excess_demand / supply))
    LOG.info(f"Max unit diff: {maxdiff}")

    # Use the approach defined in Tra (2007), page 108, eq. 7.7/7.7a, which is in turn from Anas (1982).
    alt_price = price[choiceidx]

    budget = np.zeros_like(alt_income)
    if max_rent_to_income is None:
        feasible_alts = np.full_like(alt_price, True)
    else:
        feasible_alts = alt_income * max_rent_to_income > alt_price

    budget = np.full_like(alt_income, np.nan)
    budget[feasible_alts] = price_income_transformation.apply(
        alt_income[feasible_alts], alt_price[feasible_alts], *price_income_params
    )
    budget_step = np.full_like(alt_income, np.nan)

    if max_rent_to_income is None:
        feasible_alts_step = np.full_like(alt_price, True)
    else:
        feasible_alts_step = alt_income * max_rent_to_income > alt_price + step

    budget_step[feasible_alts_step] = price_income_transformation.apply(
        alt_income[feasible_alts_step],
        alt_price[feasible_alts_step] + step,
        *price_income_params,
    )

    deriv = compute_derivatives(
        price,
        alt_income,
        choiceidx,
        hhidx,
        non_price_utilities,
        budget,
        budget_step,
        step,
        budget_coef,
        shares,
        max_rent_to_income,
        feasible_alts,
        feasible_alts_step,
        weights,
    )

    if not np.all(deriv < 0):
        raise ValueError("some derivatives of price are nonnegative")

    LOG.info(f"Computed price derivatives:\n{pd.Series(deriv).describe()}")

    # this is 7.7a from Tra's dissertation
    orig_fixed_price = price[fixed_price]
    price = price - (excess_demand / deriv) * speed_control

    # fix one price from changing so the system is defined
    price[fixed_price] = orig_fixed_price

    return price, False  # not converged yet (or we don't know anyhow)


# Numba does not help with this function
def compute_derivatives(
    price,
    alt_income,
    choiceidx,
    hhidx,
    non_price_utilities,
    budget,
    budget_step,
    price_step,
    budget_coef,
    base_shares,
    max_rent_to_income,
    feasible_alts,
    feasible_alts_step,
    weights,
):
    deriv = np.zeros_like(price)

    base_exp_utilities = np.exp(non_price_utilities + budget_coef * budget)
    # exp utility of zero is out of choice set
    base_exp_utilities[~feasible_alts] = 0

    # memmap the base utilities so we can use copy-on-write
    fh, util_file = tempfile.mkstemp(prefix="eqsormo_mmap_", suffix=".bin")
    os.close(fh)
    try:
        mm = np.memmap(
            util_file, dtype="float64", mode="w+", shape=base_exp_utilities.shape
        )
        mm[:] = base_exp_utilities[:]
        mm.flush()
        if not np.all(np.isfinite(base_exp_utilities)):
            raise FloatingPointError(
                f"Not all exp(utilities) are finite (scaling?)\n"
                f"min non-price utility: {np.min(non_price_utilities)}\n"
                f"max non-price utility: {np.max(non_price_utilities)}\n"
                f"nans in non-price utility: {np.sum(np.isnan(non_price_utilities))}\n"
                f"min budget: {np.min(budget[feasible_alts])}\n"
                f"max budget: {np.max(budget[feasible_alts])}\n"
                f"nans in budget: {np.sum(np.isnan(budget[feasible_alts]))}\n"
            )
        del base_exp_utilities

        step_exp_utilities = np.exp(non_price_utilities + budget_coef * budget_step)
        step_exp_utilities[~feasible_alts_step] = 0

        if not np.all(np.isfinite(step_exp_utilities)):
            raise FloatingPointError(
                "Not all exp(step_utilities) are finite (scaling?)\n"
                f"min non-price utility: {np.min(non_price_utilities)}\n"
                f"max non-price utility: {np.max(non_price_utilities)}\n"
                f"nans in non-price utility: {np.sum(np.isnan(non_price_utilities))}\n"
                f"min budget: {np.min(budget_step[feasible_alts_step])}\n"
                f"max budget: {np.max(budget_step[feasible_alts_step])}\n"
                f"nans in budget: {np.sum(np.isnan(budget_step[feasible_alts_step]))}\n"
            )

        # cache weights
        if weights is not None:
            alt_weights = weights[hhidx]

        task_queue = queue.Queue()
        result_queue = queue.Queue()
        stop_threads = threading.Event()

        def worker():
            while not stop_threads.is_set():
                try:
                    i = task_queue.get(timeout=10)
                except queue.Empty:
                    continue  # return to top of loop to check for stop_threads event
                else:
                    exp_utilities = np.memmap(
                        util_file,
                        dtype="float64",
                        mode="c",
                        shape=step_exp_utilities.shape,
                    )
                    # save memory by just saving indices, not full bool array
                    # in the model in my dissertation, there are ~1000 choices, so this will be 1/1000 of the indices
                    # since an int is 32 bytes (or maybe 64 since numpy arrays can be big), and a bool I believe takes up
                    # an entire byte, we come out ahead b/c 1/1000 * 32 < 8
                    choicemask = np.nonzero(choiceidx == i)
                    exp_utilities[choicemask] = step_exp_utilities[choicemask]
                    util_sums = np.bincount(hhidx, weights=exp_utilities)
                    probs = exp_utilities[choicemask] / util_sums[hhidx[choicemask]]
                    if weights is not None:
                        probs *= alt_weights[choicemask]
                    share_step = np.sum(probs)
                    del probs
                    deriv = (share_step - base_shares[i]) / price_step
                    result_queue.put((i, deriv))
                    task_queue.task_done()

        def consumer():
            while not stop_threads.is_set():
                try:
                    i, val = result_queue.get(timeout=10)
                except queue.Empty:
                    continue
                else:
                    assert (
                        val < 0
                    ), f"derivative of price for alt {i} is non-negative: {val}"
                    if i % 10 == 9:
                        LOG.info(f"computed derivative {i + 1} / {len(deriv)}")
                    deriv[i] = val
                    result_queue.task_done()

        # might need something more complex here to account for the memory pressure of sorting. Even if you have
        # 16 cores, you might not have enough memory to compute 16 derivatives at once.
        nthreads = multiprocessing.cpu_count()
        LOG.info(f"computing derivatives using {nthreads} threads")

        # start threads
        for i in range(nthreads):
            threading.Thread(target=worker, daemon=False).start()

        # consumer thread
        threading.Thread(target=consumer, daemon=False).start()

        # fill queue
        for i in range(len(price)):
            task_queue.put(i)

        # await completion
        task_queue.join()
        result_queue.join()

        # signal threads to shut down
        stop_threads.set()

        return deriv
    finally:
        # clean up
        os.remove(util_file)


def compute_shares(
    price,
    supply,
    alt_income,
    choiceidx,
    hhidx,
    non_price_utilities,
    price_income_transformation,
    price_income_params,
    budget_coef,
    max_rent_to_income,
    weights,
):
    alt_price = price[choiceidx]

    # unfortunately feasible alts does need to be recalculated on each iter as prices may change
    if max_rent_to_income is None:
        feasible_alts = np.full_like(alt_price, True)
    else:
        feasible_alts = alt_income * max_rent_to_income > alt_price

    budgets = np.zeros_like(alt_price)
    budgets[feasible_alts] = price_income_transformation.apply(
        alt_income[feasible_alts], alt_price[feasible_alts], *price_income_params
    )
    full_utilities = non_price_utilities + budget_coef * budgets
    exp_utility = np.exp(full_utilities)
    del full_utilities, budgets
    exp_utility[
        ~feasible_alts
    ] = 0  # will force choice probability to zero for infeasible alts
    del feasible_alts

    if not np.all(np.isfinite(exp_utility)):
        raise FloatingPointError("Not all exp(utilities) are finite (scaling?)")

    expsums = np.bincount(hhidx, weights=exp_utility)
    probs = exp_utility / expsums[hhidx]

    if weights is not None:
        probs *= weights[hhidx]

    return np.bincount(choiceidx, weights=probs)
