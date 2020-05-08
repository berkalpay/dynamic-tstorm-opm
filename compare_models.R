library(hydroGOF)
library(xtable)

mape <- function(true, pred) {
  mean( abs((true-pred)/true) )
}
t_m <- function(series) {
  return( sum( (1:length(series)) * series ) / sum(series) )
}
t_var <- function(series) {
  diff <- (1:length(series)) - t_m(series)
  return( sum((1:(length(series)))^2 * series)/sum(series) - t_m(series)^2 )
}
sod <- function(v) sum(v>10)

pred_df <- read.csv("results/predictions.csv")
an_df <- read.csv("data/storm_hours_outages.csv")
models <- c("eopm", "knn", "lstm", "rf", "glm")

# Pan-model comparisons
metrics <- c("t_m MAE","t_var MAE",
             "r^2 SOD","r^2 hourly","NSE hourly",
             "r^2 storm","NSE storm","MAPE storm")
metrics_df <- data.frame(metrics=metrics)
for(model in models) {
  # Hourly timeseries metrics
  pred_tm <- aggregate(pred_df[,model] ~ pred_df$sid, FUN=t_m)[,2]
  pred_tvar <- aggregate(pred_df[,model] ~ pred_df$sid, FUN=t_var)[,2]
  pred_sod <- aggregate(pred_df[,model] ~ pred_df$sid, FUN=sod)[,2]
  act_tm <- aggregate(pred_df$countts ~ pred_df$sid, FUN=t_m)[,2]
  act_tvar <- aggregate(pred_df$countts ~ pred_df$sid, FUN=t_var)[,2]
  act_sod <- aggregate(pred_df$countts ~ pred_df$sid, FUN=sod)[,2]
  
  # Hourly vector metrics
  h_cor <- c()
  h_nse <- c()
  h_mape <- c()
  for(i in unique(pred_df$sid)) {
    pred_df_i <- pred_df[pred_df$sid==i,]
    pred_i <- pred_df_i[,model]
    act_i <- pred_df_i$countts
    
    h_cor <- c(h_cor, cor(act_i, pred_i)^2)
    h_nse <- c(h_nse, NSE(pred_i, act_i))
    h_mape <- c(h_mape, mape(act_i, pred_i))
  }
  
  # Storm total metrics
  pred_storm_tots <- aggregate(pred_df[,model] ~ pred_df$sid, FUN=sum)[,2]
  act_storm_tots <- aggregate(an_df$countts ~ pred_df$sid, FUN=sum)[,2]
  
  metrics_df[,model] <- c(mean(abs(act_tm - pred_tm)),
                          mean(abs(act_tvar - pred_tvar)),
                          cor(act_sod, pred_sod)^2,
                          mean(h_cor),
                          mean(h_nse),
                          cor(pred_storm_tots, act_storm_tots)^2,
                          NSE(pred_storm_tots, act_storm_tots),
                          mape(act_storm_tots, pred_storm_tots))
}

# Output as LaTeX table
rownames(metrics_df) <- metrics_df$metrics
metrics_df <- metrics_df[,-1]
metrics_df <- t(metrics_df)
rownames(metrics_df) <- toupper(models)
xtable(metrics_df, digits=3)
