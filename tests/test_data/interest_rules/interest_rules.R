library(lpirfs)

full = interest_rules_var_data

endog = full[c("GDP_gap")]
shock = full[c("Infl")]

res = lp_lin_iv(endog=endog, shock=shock, lags_endog_lin=0, trend=0, confint=1, hor=13, use_nw=FALSE)

mean = res$irf_lin_mean[1, ]
upper = res$irf_lin_up[1, ]
lower = res$irf_lin_low[1, ]

output = data.frame(mean=mean, upper=upper, lower=lower)
write.csv(output, "./interest_rules_no_control.csv", row.names=FALSE)


res = lp_lin_iv(endog=endog, shock=shock, lags_endog_lin=0, trend=0, confint=1, hor=13, use_nw=TRUE)

mean = res$irf_lin_mean[1, ]
upper = res$irf_lin_up[1, ]
lower = res$irf_lin_low[1, ]

output = data.frame(mean=mean, upper=upper, lower=lower)
write.csv(output, "./interest_rules_no_control_nw.csv", row.names=FALSE)


res = lp_lin_iv(endog=endog, shock=shock, lags_endog_lin=4, trend=0, confint=1, hor=13, use_nw=FALSE)

mean = res$irf_lin_mean[1, ]
upper = res$irf_lin_up[1, ]
lower = res$irf_lin_low[1, ]

output = data.frame(mean=mean, upper=upper, lower=lower)
write.csv(output, "./interest_rules_no_control_lags4.csv", row.names=FALSE)


res = lp_lin_iv(endog=endog, shock=shock, lags_endog_lin=4, trend=0, confint=1, hor=13, use_nw=TRUE)

mean = res$irf_lin_mean[1, ]
upper = res$irf_lin_up[1, ]
lower = res$irf_lin_low[1, ]

output = data.frame(mean=mean, upper=upper, lower=lower)
write.csv(output, "./interest_rules_no_control_nw_lags4.csv", row.names=FALSE)
