library(lpirfs)

full = ag_data[ag_data["Year"] >= 1950, ]
full = full[2:nrow(full), ]
endog = full[c("GDP")]
shock = full[c("Gov_shock_mean")]

res = lp_lin_iv(endog=endog, shock=shock, lags_endog_lin=0, trend=0, confint=1, hor=13, use_nw=TRUE)

mean = res$irf_lin_mean[1, ]
upper = res$irf_lin_up[1, ]
lower = res$irf_lin_low[1, ]

output = data.frame(mean=mean, upper=upper, lower=lower)
write.csv(output, "./ag_data_irf_nw.csv", row.names=FALSE)


res = lp_lin_iv(endog=endog, shock=shock, lags_endog_lin=0, trend=0, confint=1, hor=13, use_nw=FALSE)

mean = res$irf_lin_mean[1, ]
upper = res$irf_lin_up[1, ]
lower = res$irf_lin_low[1, ]

output = data.frame(mean=mean, upper=upper, lower=lower)
write.csv(output, "./ag_data_irf.csv", row.names=FALSE)
