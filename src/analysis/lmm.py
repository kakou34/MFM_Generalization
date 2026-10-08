"""Linear mixed models through R (lme4 / lmerTest / performance) via rpy2: REML, Satterthwaite t-tests, Nakagawa R2.
Same engine as correlate_new.ipynb."""
from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib import callbacks as _cb
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy import stats

_cb.consolewrite_warnerror = lambda x: None  # silence R's package-load chatter (errors still raise)

ro.r("""
suppressPackageStartupMessages({library(lme4); library(lmerTest); library(performance)})
.fit_lmm <- function(dat, formula) {
  fit <- suppressWarnings(suppressMessages(lmerTest::lmer(as.formula(formula), data = dat)))
  fixed <- as.data.frame(coef(summary(fit)))
  fixed <- data.frame(term = rownames(fixed), fixed, row.names = NULL, check.names = FALSE)
  colnames(fixed) <- c("term", "estimate", "se", "df", "t", "p")
  r2 <- tryCatch(suppressWarnings(performance::r2_nakagawa(fit)), error = function(e) NULL)
  list(fixed = fixed,
       info = data.frame(r2_marginal = if (is.null(r2)) NA_real_ else as.numeric(r2$R2_marginal),
                         r2_conditional = if (is.null(r2)) NA_real_ else as.numeric(r2$R2_conditional),
                         nobs = nobs(fit)))
}
""")

RE = "(1|model) + (1|train) + (1|test)"


def _df(obj) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(obj)


def fit_lmm(data: pd.DataFrame, formula: str) -> dict:
    """Fit ``formula``; model / train / test are cast to factors. Returns {'fixed': DataFrame, 'info': dict}."""
    d = data.copy()
    for g in ("model", "train", "test"):
        d[g] = d[g].astype(str)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(d)
    out = ro.r[".fit_lmm"](r_df, formula)
    fixed = _df(out.rx2("fixed"))
    t = stats.t.ppf(0.975, fixed["df"].clip(lower=1))
    fixed["ci_low"], fixed["ci_high"] = fixed["estimate"] - t * fixed["se"], fixed["estimate"] + t * fixed["se"]
    return {"fixed": fixed, "info": _df(out.rx2("info")).iloc[0].to_dict()}


def term_row(res: dict, term: str) -> dict:
    """One line for a fixed term: estimate, SE, Satterthwaite df, p, 95% CI, R2 of the whole model."""
    r = res["fixed"].set_index("term").loc[term]
    return {"estimate": r["estimate"], "se": r["se"], "df": r["df"], "p": r["p"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "r2_marginal": res["info"]["r2_marginal"],
            "r2_conditional": res["info"]["r2_conditional"], "n": int(res["info"]["nobs"])}


def main_lmm(d: pd.DataFrame) -> dict:
    """auc_disease ~ auc_cohort + (1|model) + (1|train) + (1|test) -> row for the auc_cohort slope."""
    return term_row(fit_lmm(d, f"auc_disease_mean ~ auc_cohort_mean + {RE}"), "auc_cohort_mean")
