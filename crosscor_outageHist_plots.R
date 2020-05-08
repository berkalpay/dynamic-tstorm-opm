df <- read.csv("data/Calib_hrrrHourly_summerCT_townFeat_totCounts.csv", row.names=1)

df_winds <- df[startsWith(names(df), "WIND")]
mean_winds <- apply(df_winds, MARGIN=1, mean)

pdf("plots/wind_lag_crosscorr_outage_distr.pdf",
    width=8, height=4)
  par(mfrow=c(1,2), cex.lab=1.26)
  ccf(mean_winds, df$countts, xlim=c(-7, 10),
      xlab="Lag (hours)", ylab="Cross-correlation", main="")
  hist(df$countts,
       xlab="Outages per hour", main="")
dev.off()
