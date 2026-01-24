# ============================================================
# Mixed-effects model in R (lme4) matching your Python workflow
# - Reads df from CSV
# - Factors: inout, size, probe, subject
# - Sets probe baseline to "G" (if present)
# - Fits: rt ~ predicted + inout + size + probe + (1|subject)
# - Outputs a clean fixed-effects table: Coef, CI, t, p, p-FDR
# ============================================================

# ---- Install (run once) ----
# install.packages(c("lme4", "lmerTest", "readr"))

# ---- Load ----
library(lme4)
library(lmerTest)
library(readr)

# ---- Read CSV ----
# Put your df.csv in the working directory, or provide a full path.
df <- read_csv("E:\\Compositional_encoding\\Code\\table_df.csv")

# ---- Make sure categorical variables are factors ----
df$inout   <- factor(df$inout)     # e.g., 51/52
df$size    <- factor(df$size)      # e.g., 4/6/8
df$probe   <- factor(df$probe)     # letters
df$subject <- factor(df$subject)   # subject IDs

df$inout <- factor(df$inout, levels = rev(levels(df$inout)))
df$size  <- factor(df$size,  levels = rev(levels(df$size)))

# ---- Set 'G' as the reference (baseline) level for probe ----
df$probe <- droplevels(df$probe)
if ("G" %in% levels(df$probe)) {
  df$probe <- relevel(df$probe, ref = "G")
} else {
  warning("'G' is not present in probe categories; default reference will be used.")
}

# ---- Fit mixed model (random intercept for subject), ML like Python reml=False ----
m <- lmer(
  rt ~ predicted + inout + size + probe + (1 | subject),
  data = df,
  REML = FALSE
)

# Optional: print model summary to console
print(summary(m))

# ============================================================
# Build clean fixed-effects table: Coef, CI [low, high], t, p
# (lmerTest gives t + p-values; Python MixedLM reports z by default)
# ============================================================

s <- summary(m)
coefs <- s$coefficients
# columns typically: Estimate, Std. Error, df, t value, Pr(>|t|)

# Wald CIs for fixed effects (like a quick CI from standard errors)
ci <- confint(m, parm = "beta_", method = "Wald")  # only fixed effects

results_table <- data.frame(
  term = rownames(coefs),
  Coefficient = round(coefs[, "Estimate"], 3),
  CI = sprintf("[%.3f, %.3f]", ci[, 1], ci[, 2]),
  t = round(coefs[, "t value"], 3),
  p_value = format.pval(coefs[, "Pr(>|t|)"], digits = 2, eps = 1e-99),
  stringsAsFactors = FALSE
)

# ============================================================
# FDR (BH) correction separately for probe and size terms
# ============================================================

probe_mask <- grepl("^probe", results_table$term)
size_mask  <- grepl("^size",  results_table$term)

results_table$p_FDR <- ""

results_table$p_FDR[probe_mask] <- format.pval(
  p.adjust(coefs[probe_mask, "Pr(>|t|)"], method = "BH"),
  digits = 2, eps = 1e-99
)

results_table$p_FDR[size_mask] <- format.pval(
  p.adjust(coefs[size_mask, "Pr(>|t|)"], method = "BH"),
  digits = 2, eps = 1e-99
)

# ---- View results ----
print(results_table, row.names = FALSE)

results_table <- data.frame(
  term = rownames(coefs),
  Coefficient = round(coefs[, "Estimate"], 3),
  CI = sprintf("[%.3f, %.3f]", ci[, 1], ci[, 2]),
  t = round(coefs[, "t value"], 3),
  p_raw = coefs[, "Pr(>|t|)"],
  stringsAsFactors = FALSE
)

probe_mask <- grepl("^probe", results_table$term)
size_mask  <- grepl("^size",  results_table$term)

results_table$p_FDR <- NA_real_

results_table$p_FDR[probe_mask] <- p.adjust(
  results_table$p_raw[probe_mask], method = "BH"
)

results_table$p_FDR[size_mask] <- p.adjust(
  results_table$p_raw[size_mask], method = "BH"
)

results_table$p_value <- ifelse(
  is.na(results_table$p_FDR),
  results_table$p_raw,
  results_table$p_FDR
)

# Pretty formatting
results_table$p_value <- format.pval(
  results_table$p_value, digits = 2, eps = 1e-99
)

results_table$p_raw <- NULL
results_table$p_FDR <- NULL


# ---- Save results to CSV (optional) ----
write.csv(results_table, "fixed_effects_results_table.csv", row.names = FALSE)
