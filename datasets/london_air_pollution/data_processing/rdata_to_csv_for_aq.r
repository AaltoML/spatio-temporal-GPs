#!/usr/bin/env Rscript


args = commandArgs(trailingOnly=TRUE)

rdata_input = args[1]
csv_output = args[2]
df = args[3]

load(rdata_input)

options(TZ="Europe/London")
Sys.setenv (TZ="Europe/London")

print(rdata_input)
print(csv_output)

#a = subset(get(df), select=-c(SiteName, Address, Authority))
a = get(df)
write.table(a, file=csv_output, row.names = FALSE, sep=';', na='')
