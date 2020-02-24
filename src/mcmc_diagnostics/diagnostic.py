import arviz as az
from arviz.stats import ess
import numpy as np

# plot settings
az.style.use('arviz-darkgrid')


class MCMCDiagnostic(object):

    def __init__(self, chains_dict):
        self.chains = az.convert_to_inference_data(chains_dict)

    def get_chain(self, chain_id):
        return self.chains.posterior[chain_id].values

    def ess_plot(self, chain_id, num_points=10):
        plts = az.plot_ess(self.chains, var_names=chain_id, kind='evolution', min_ess=0)
        for i in range(len(plts)):
            plot = plts[i]
            plot.axes.lines[1].remove()
            plot.axes.get_legend().remove()
            y_data = plot.axes.lines[0].get_ydata()
            y_max = np.nanmax(y_data)
            plot.axes.set_ylim(0, y_max + 10.)
            plts[i] = plot

        return plts

    def trace_plot(self, chain_id):
        return az.plot_trace(self.chains, var_names=chain_id)

    def autocorr_plot(self, chain_id, **kwargs):
        return az.plot_autocorr(self.chains, var_names=chain_id,  **kwargs)

    def ess(self, chain_id):
        chain = self.get_chain(chain_id)
        chain = chain[~np.isnan(chain)]
        return self.ess_(chain)

    def ess_(self, chain):
        return ess(chain)

