import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

class LinearRegDiagnostic:
    """
    Generates full regression diagnostic plots for a fitted OLS model.
    Includes:
    1. Residuals vs Fitted values
    2. Standardized Residuals vs Theoretical Quantiles (QQ plot)
    3. Sqrt(Standardized Residuals) vs Fitted values
    4. Residuals vs Leverage
    5. Histogram of residuals
    """

    def __init__(self, model):
        self.model = model

    def plot_all(self, filename_prefix="UE_04_App2_DiagnosticPlots"):
        # Residuals vs Fitted
        fig1 = plt.figure(figsize=(6,5))
        fitted_vals = self.model.fittedvalues
        residuals = self.model.resid
        ax1 = fig1.add_subplot(111)
        ax1.scatter(fitted_vals, residuals, edgecolors='k', facecolors='none')
        ax1.axhline(y=0, linestyle='--', color='red')
        ax1.set_xlabel('Fitted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted values')
        fig1.tight_layout()
        fig1.savefig(f"{filename_prefix}_ResidualsVsFitted.pdf")
        plt.close(fig1)

        # QQ Plot
        fig2 = plt.figure(figsize=(6,5))
        ax2 = fig2.add_subplot(111)
        sm.qqplot(self.model.get_influence().resid_studentized_internal, line='45', ax=ax2)
        ax2.set_title('Standardized Residuals vs Theoretical Quantiles')
        fig2.tight_layout()
        fig2.savefig(f"{filename_prefix}_QQPlot.pdf")
        plt.close(fig2)

        # Sqrt(Standardized Residuals) vs Fitted values
        fig3 = plt.figure(figsize=(6,5))
        ax3 = fig3.add_subplot(111)
        standardized_resid = self.model.get_influence().resid_studentized_internal
        ax3.scatter(fitted_vals, np.sqrt(np.abs(standardized_resid)), edgecolors='k', facecolors='none')
        ax3.set_xlabel('Fitted values')
        ax3.set_ylabel('Sqrt(Standardized Residuals)')
        ax3.set_title('Sqrt(Standardized Residuals) vs Fitted values')
        fig3.tight_layout()
        fig3.savefig(f"{filename_prefix}_SqrtStdResidVsFitted.pdf")
        plt.close(fig3)

        # Residuals vs Leverage (Influence plot)
        fig4 = plt.figure(figsize=(6,5))
        ax4 = fig4.add_subplot(111)
        sm.graphics.influence_plot(self.model, ax=ax4, criterion="cooks")
        ax4.set_title('Residuals vs Leverage')
        fig4.tight_layout()
        fig4.savefig(f"{filename_prefix}_ResidualsVsLeverage.pdf")
        plt.close(fig4)

        # Histogram of residuals
        fig5 = plt.figure(figsize=(6,5))
        ax5 = fig5.add_subplot(111)
        ax5.hist(residuals, bins=20, edgecolor='k')
        ax5.set_title('Histogram of Residuals')
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        fig5.tight_layout()
        fig5.savefig(f"{filename_prefix}_HistogramResiduals.pdf")
        plt.close(fig5)

        print("Diagnostic plots generated: ResidualsVsFitted, QQPlot, SqrtStdResidVsFitted, ResidualsVsLeverage, HistogramResiduals")
