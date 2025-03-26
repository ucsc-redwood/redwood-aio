import pandas as pd
import io

data_str = """\
Device     App        Schedule ID  Max Chunk Time  Avg Time/Task   Difference %
3A021JEHN02756 CifarSparse 034          3.02            3.02            0.06
3A021JEHN02756 CifarSparse 010          2.42            2.43            0.51
3A021JEHN02756 CifarSparse 044          3.02            3.04            0.79
3A021JEHN02756 CifarSparse 036          3.02            3.06            1.30
3A021JEHN02756 CifarSparse 022          2.59            2.54            2.13
3A021JEHN02756 CifarSparse 040          3.02            2.94            2.67
3A021JEHN02756 CifarSparse 047          3.02            2.92            3.30
3A021JEHN02756 CifarSparse 023          2.59            2.68            3.35
3A021JEHN02756 CifarSparse 015          2.42            2.30            4.90
3A021JEHN02756 CifarSparse 024          2.59            2.44            5.69
3A021JEHN02756 CifarSparse 048          3.02            2.85            5.76
3A021JEHN02756 CifarSparse 021          2.58            2.43            5.79
3A021JEHN02756 CifarSparse 043          3.02            2.84            5.89
3A021JEHN02756 CifarSparse 018          2.42            2.27            6.34
3A021JEHN02756 CifarSparse 027          2.99            3.21            7.40
3A021JEHN02756 CifarSparse 038          3.02            2.75            8.95
3A021JEHN02756 CifarSparse 033          3.02            2.75            9.09
3A021JEHN02756 CifarSparse 035          3.02            2.71            10.23
3A021JEHN02756 CifarSparse 011          2.42            2.71            11.73
3A021JEHN02756 CifarSparse 012          2.42            2.74            13.25
3A021JEHN02756 CifarSparse 009          2.42            2.77            14.22
3A021JEHN02756 CifarSparse 013          2.42            2.78            14.93
3A021JEHN02756 CifarSparse 042          3.02            3.52            16.43
3A021JEHN02756 CifarSparse 020          2.42            2.85            17.55
3A021JEHN02756 CifarSparse 041          3.02            3.60            19.07
3A021JEHN02756 CifarSparse 045          3.02            3.60            19.29
3A021JEHN02756 CifarSparse 039          3.02            3.67            21.59
3A021JEHN02756 CifarSparse 046          3.02            3.68            21.98
3A021JEHN02756 CifarSparse 014          2.42            3.00            23.78
3A021JEHN02756 CifarSparse 049          3.02            3.82            26.36
3A021JEHN02756 CifarSparse 017          2.42            3.07            26.65
3A021JEHN02756 CifarSparse 037          3.02            3.83            26.68
3A021JEHN02756 CifarSparse 019          2.42            3.16            30.31
3A021JEHN02756 CifarSparse 050          3.02            3.94            30.34
3A021JEHN02756 CifarSparse 028          2.99            3.90            30.46
3A021JEHN02756 CifarSparse 026          2.63            3.48            32.50
3A021JEHN02756 CifarSparse 025          2.63            3.49            32.78
3A021JEHN02756 CifarSparse 016          2.42            3.30            36.46
3A021JEHN02756 CifarSparse 008          2.04            2.93            44.14
3A021JEHN02756 CifarSparse 029          2.99            4.38            46.68
3A021JEHN02756 CifarSparse 006          1.84            2.80            52.26
3A021JEHN02756 CifarSparse 031          2.99            4.60            53.58
3A021JEHN02756 CifarSparse 003          1.84            2.85            55.32
3A021JEHN02756 CifarSparse 032          2.99            4.66            55.74
3A021JEHN02756 CifarSparse 030          2.99            4.67            56.27
3A021JEHN02756 CifarSparse 004          1.84            2.90            57.88
3A021JEHN02756 CifarSparse 007          1.94            3.08            58.61
3A021JEHN02756 CifarSparse 005          1.84            2.98            62.42
3A021JEHN02756 CifarSparse 001          1.84            3.00            63.34
3A021JEHN02756 CifarSparse 002          1.84            3.13            70.67
"""

df = pd.read_csv(
    io.StringIO(data_str),
    delim_whitespace=True,
    names=[
        "Device",
        "App",
        "Schedule_ID",
        "Max_Chunk_Time",
        "Avg_Time_Task",
        "Diff_Percent",
    ],
    skiprows=1,
)

df.head()


corr_pearson = df['Max_Chunk_Time'].corr(df['Avg_Time_Task'], method='pearson')
corr_pearson


corr_spearman = df['Max_Chunk_Time'].corr(df['Avg_Time_Task'], method='spearman')
corr_spearman


import statsmodels.api as sm

X = df['Max_Chunk_Time']
y = df['Avg_Time_Task']

# Add a constant term so we fit intercept as well
X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const).fit()
print(model.summary())


import matplotlib.pyplot as plt

plt.scatter(df['Max_Chunk_Time'], df['Avg_Time_Task'])
plt.xlabel("Max Chunk Time (Hypothesized)")
plt.ylabel("Avg Time/Task (Measured)")
plt.title("Scatter Plot: Hypothesized vs. Measured Time")
plt.show()

