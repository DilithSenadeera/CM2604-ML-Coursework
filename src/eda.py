"""Audit the source data and plot churn patterns in the training sample."""

from hashlib import sha256

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from churn_data import DATA_FILE, NUMERIC_COLUMNS, PROJECT_ROOT, load_data, split_data


FIGURE_FILE = PROJECT_ROOT / "results" / "training-eda.png"
COLORS = ["#246674", "#a84b27"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]


def make_figure(train, contract_summary):
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), layout="constrained")
        figure.suptitle(f"Telco customer churn in the training sample (n = {len(train):,})",
                        fontsize=17, fontweight="bold")
        share = train["Churn"].mean()
        counts = train["Churn"].value_counts().reindex([0, 1])

        target = axes[0, 0]
        bars = target.bar(["Stayed", "Churned"], counts / len(train), color=COLORS, width=0.60)
        target.bar_label(bars, [f"{n / len(train):.1%}\n(n={n:,})" for n in counts], padding=4)
        target.set(title="Churn is the smaller class", ylabel="Share of customers", ylim=(0, 0.91))
        target.yaxis.set_major_formatter(PercentFormatter(1))

        contract = axes[0, 1]
        bars = contract.bar(CONTRACTS, contract_summary["rate"],
                            color=COLORS[0], width=0.60)
        contract.bar_label(
            bars,
            [f"{row.rate:.1%}\n(n={row.customers:,})" for row in contract_summary.itertuples()],
            padding=4,
        )
        contract.axhline(share, color="#626a73", linestyle="--", linewidth=1,
                         label=f"Overall rate ({share:.1%})")
        contract.legend(loc="upper right", frameon=False)
        contract.set(title="Churn rate by contract", ylabel="Share who churned", ylim=(0, 0.63))
        contract.yaxis.set_major_formatter(PercentFormatter(1))

        for axis, column, title, units in [
            (axes[1, 0], "tenure", "Tenure by churn status", "Months"),
            (axes[1, 1], "MonthlyCharges", "Monthly charges by churn status", "Monthly charge"),
        ]:
            groups = [train.loc[train["Churn"].eq(label), column] for label in (0, 1)]
            boxes = axis.boxplot(groups, tick_labels=["Stayed", "Churned"], widths=0.48,
                                 patch_artist=True, medianprops={"color": "#ffffff", "linewidth": 2})
            for box, color in zip(boxes["boxes"], COLORS):
                box.set(facecolor=color, edgecolor=color)
            axis.set(title=title, ylabel=units)

        FIGURE_FILE.parent.mkdir(exist_ok=True)
        figure.savefig(FIGURE_FILE, dpi=180, facecolor="white")
        plt.close(figure)


def main():
    customers, blank_total = load_data()
    train, test = split_data(customers)
    if len(train) + len(test) != len(customers) or set(train["customerID"]) & set(test["customerID"]):
        raise ValueError("The split does not partition distinct customer records.")

    print(f"CSV SHA-256: {sha256(DATA_FILE.read_bytes()).hexdigest()}")
    print(f"Shape: {customers.shape}. Unique IDs: {customers['customerID'].nunique()}.")
    print(f"Blank TotalCharges at import: {blank_total}. Zero-tenure rows: {customers['tenure'].eq(0).sum()}.")
    print(f"Repeated complete rows: {customers.duplicated().sum()}.")
    no_internet = customers["InternetService"] == "No"
    inconsistent_addons = 0
    for column in INTERNET_ADDONS:
        no_addon = customers[column] == "No internet service"
        inconsistent_addons += (no_addon != no_internet).sum()

    no_phone = customers["PhoneService"] == "No"
    no_multiple_lines = customers["MultipleLines"] == "No phone service"
    inconsistent_lines = (no_multiple_lines != no_phone).sum()
    print(f"Inconsistent internet add-on states: {inconsistent_addons}.")
    print(f"Inconsistent phone-line states: {inconsistent_lines}.")
    print(f"Dataset target totals: {customers['Churn'].value_counts().sort_index().to_dict()}")
    print(f"Training class totals: {train['Churn'].value_counts().sort_index().to_dict()}")
    print(f"Split: {len(train)} training, {len(test)} reserved.")

    contract = train.groupby("Contract")["Churn"].agg(customers="size", churned="sum", rate="mean")
    contract = contract.loc[CONTRACTS]
    print("\nTraining churn by contract")
    print(contract.round(4).to_string())
    print("\nTraining numeric summary")
    print(train[NUMERIC_COLUMNS].describe().loc[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
    print("\nTraining medians by churn label")
    print(train.groupby("Churn")[NUMERIC_COLUMNS].median().round(2).to_string())
    print(f"Training Pearson r (tenure, total charges): {train['tenure'].corr(train['TotalCharges']):.3f}")
    make_figure(train, contract)


if __name__ == "__main__":
    main()
