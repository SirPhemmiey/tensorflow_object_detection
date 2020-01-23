

with open("data.csv", "r") as f:
    lines = f.readlines()
with open("data.csv", "w") as f:
    for line in lines:
        if line.strip("\n") != "start,end,prediction,probability,inference_time,humans,trucks,total_cycles,avg_cycle_time,avg_fill_rate":
            f.write(line)