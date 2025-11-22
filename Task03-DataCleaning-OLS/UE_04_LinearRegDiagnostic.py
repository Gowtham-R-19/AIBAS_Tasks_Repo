import matplotlib.pyplot as plt
import statsmodels.api as sm

class LinearRegDiagnostic:
    """
    Generates linear regression diagnostic plots for a fitted OLS model.
    """

    def __init__(self, model):
        self.model = model

    def plot_all(self, filename="UE_04_App2_DiagnosticPlots.pdf"):
        fig = plt.figure(figsize=(12, 8))

        # Residuals vs Fitted
        ax1 = fig.add_subplot(221)
        fitted_vals = self.model.fittedvalues
        residuals = self.model.resid
        ax1.scatter(fitted_vals, residuals, edgecolors='k', facecolors='none')
        ax1.axhline(y=0, linestyle='--', color='red')
        ax1.set_xlabel('Fitted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')

        # QQ Plot
        ax2 = fig.add_subplot(222)
        sm.qqplot(residuals, line='45', ax=ax2)
        ax2.set_title('QQ Plot of Residuals')

        # Influence Plot
        ax3 = fig.add_subplot(223)
        sm.graphics.influence_plot(self.model, ax=ax3, criterion="cooks")

        # Histogram of residuals
        ax4 = fig.add_subplot(224)
        ax4.hist(residuals, bins=20, edgecolor='k')
        ax4.set_title('Histogram of Residuals')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
