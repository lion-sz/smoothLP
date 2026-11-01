using LocalProjections, Parquet, DataFrames, CSV

dat = read_parquet("sales.parquet")


g = exp.(LinRange(1, 20, 1000))
est = SmoothLP(:MP_median, 3, 2, search=grid(g), criterion=GCV())
vce = HARVCE(EqualWeightedCosine())
model = lp(est, dat, :corrected, xnames=(:MP_median, :New_cases, :StringencyIndex_Average), nlag=90, nhorz=90, vce=vce)
result = irf(model, :corrected, :MP_median)

dat = DataFrame(coef=result.B, se=result.SE)
CSV.write("sales_irf_tuned.csv", dat)

model2 = lp(model, 1000.0, vce=vce)
result2 = irf(model2, :corrected, :MP_median)

dat = DataFrame(coef=result2.B, se=result2.SE)
CSV.write("sales_irf_1k.csv", dat)

