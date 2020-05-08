lag_df <- read.csv("results/find_lag_GLM.csv")
knn_df <- read.csv("results/knn_search.csv")
rf_df <- read.csv("results/rf_search.csv")

pdf("plots/lag_knn_rf_selection.pdf",
    width=12, height=4)
par(mfrow=c(1,3), cex.lab=2, cex.axis=1.5,
    mar=c(4.5,4.9,1,1))
# Lag selection
with(lag_df, plot(lag, totals_R2, type="o", col="black",
                  ylim=c(0,0.6), pch=16, lty=1,
                  xlab="Lag (hours)", ylab=expression(paste("r"^"2"))))
with(lag_df, lines(lag, hourly_R2, type="o", lty=1,
                   col="blue", pch=16))
abline(v=1, lty=2, lwd=1.2)
legend("topright", legend=c("Hourly", "Storm totals"),
       fill=c("black", "blue"), cex=1.5)
# KNN selection
with(knn_df, plot(k, hourly_R2, type="o", lty=1,
                  pch=16, cex=0.76, ylim=c(0.17,0.28),
                  xlab="k (neighbors)", ylab=""))
abline(v=74, lty=2, lwd=1.2)
# RF selection
with(rf_df, plot(ntree, hourly_R2, type="o", lty=1,
                 pch=16, ylim=c(0.25,0.4),
                 xlab="Number of trees", ylab=""))
abline(v=150, lty=2, lwd=1.2)
dev.off()
