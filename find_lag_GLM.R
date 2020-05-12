df_list <- lapply(list.files("data/LOSO/", full.names=T), read.csv, row.names=1)
an_df <- read.csv("data/storm_hours_outages.csv")
results_path <- "results/find_lag_GLM.csv"

#################
## DEFINE LAGS ##
#################

lagDF <- function(d, lag) {
  if (lag == 0)
    return(d)
  
  d_lagged <- d[(lag+1):nrow(d),]
  for(j in 1:ncol(d)) {
    col <- d[,j]
    for(l in 1:lag) {
      col_lagged <- col[l:(l+nrow(d_lagged)-1)]
      d_lagged[,paste0(names(d)[j],"_l",l)] <- col_lagged
    }
  }
  
  return(d_lagged)
}

lagOutageDF <- function(d, lag) {
  countts <- d$countts[(lag+1):length(d$countts)]
  return(cbind(lagDF(d[,-ncol(d)], lag), countts))
}

##############
## FIND LAG ##
##############

storm_id <- (0:485)%/%18

tot_cors <- c()
hourly_cors <- c()
for(lag in 0:10) {
  
  storm_pred <- c()
  for(i in 0:26) {
    dfl <- lagOutageDF(df_list[[i+1]], lag)
    
    storm_hours <- an_df$dateTime[(i*18+1):((i+1)*18)]
    
    fit <- glm(countts ~., dfl[!(rownames(dfl) %in% storm_hours),], family="poisson")
    storm_pred <- c(storm_pred, predict(fit, dfl[rownames(dfl) %in% storm_hours,],
                                        type="response"))
  }  
  pred_df <- data.frame(act=dfl$countts[rownames(dfl) %in% an_df$dateTime],
                        pred=storm_pred)
  
  pred_storm_tots <- with(pred_df, aggregate(pred ~ storm_id, FUN=sum))
  act_storm_tots <- with(pred_df, aggregate(act ~ storm_id, FUN=sum))
  tot_cors <- c(tot_cors, cor(pred_storm_tots[,2], act_storm_tots[,2])^2)
  
  p_cor <- c()
  for (i in 0:26) {
    df_i <- pred_df[storm_id==i,]
    p_cor <- c(p_cor, with(df_i, cor(act, pred))^2)
  }
  hourly_cors <- c(hourly_cors, mean(p_cor))

  print(paste(hourly_cors[lag+1], tot_cors[lag+1]))
}
  
lag_results <- data.frame(lag=0:10, hourly_R2=hourly_cors, totals_R2=tot_cors)
write.csv(lag_results, results_path, row.names=F)
