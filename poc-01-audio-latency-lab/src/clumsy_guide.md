# Using Clumsy for Network Degradation on Windows

Clumsy is the Windows equivalent of Linux `tc/netem`.
Download from: https://jagt.github.io/clumsy/

## Steps

1. Download `clumsy-0.3-win64-a.zip` from the site above
2. Extract the zip — you get a single folder
3. Right-click `clumsy.exe` → **Run as administrator**

## Settings for PoC-01

| Test scenario        | Lag   | Inbound Loss | Jitter |
|----------------------|-------|--------------|--------|
| Mild degradation     | 10ms  | 5%           | 0      |
| Moderate degradation | 20ms  | 10%          | 5ms    |
| Severe degradation   | 40ms  | 20%          | 10ms   |

## How to use

1. Open Clumsy as Administrator
2. Tick **Inbound** and **Outbound** checkboxes
3. Set your Lag and Loss values from the table above
4. Click **Start**
5. Run `latency_measure.py` in your terminal
6. Click **Stop** in Clumsy when done

## Important

Always click **Stop** in Clumsy before running the baseline measurement.
Clumsy affects all network traffic on your machine while active — including
your browser. Stop it when you are done with the experiment.