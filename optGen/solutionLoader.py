import pickle as pkl
import numpy as np


"""Reads a solution pickle and calculate the x,u,f from *_plots
"""
class SolLoader(dict):
    def __init__(self, solution_file):
        super().__init__([])
        self.solution_file = solution_file
        with open(solution_file, "rb") as f:
            self.sol = pkl.load(f)['sol']
        self.update(self.sol)
        print(self.keys())
        self.t_plot = np.row_stack( [self.sol["dTgen"]["t_plot"], 999]).reshape(-1)
        self.x_plot = np.row_stack( [self.sol["Xgen"]["x_plot"].T, self.sol["Xgen"]["x_plot"][:,-1].T])
        self.u_plot = np.row_stack( [self.sol["Ugen"]["u_plot"].T, self.sol["Ugen"]["u_plot"][:,-1].T])
        self.F_plot = np.row_stack( [self.sol["Fgen"]["F_plot"].T, self.sol["Fgen"]["F_plot"][:,-1].T])

    def itp(self,t):
        """interpolate using time stamp t
        """
        # ind = 0
        ind = np.searchsorted(self.t_plot, t, "right")

        alpha = (t - self.t_plot[ind-1])/(self.t_plot[ind] - self.t_plot[ind-1])
        # print((1-alpha)*self.t_plot[ind-1] + alpha*self.t_plot[ind])
        return {
            "t": (1-alpha)*self.t_plot[ind-1] + alpha*self.t_plot[ind],
            "u": (1-alpha)*self.u_plot[ind-1] + alpha*self.u_plot[ind],
            "x": (1-alpha)*self.x_plot[ind-1] + alpha*self.x_plot[ind],
            "F": (1-alpha)*self.F_plot[ind-1] + alpha*self.F_plot[ind]
        }
        

if __name__ == "__main__":
    sll = SolLoader("/home/ami/ycy/JyTo/data/nlpSol/sideFlip1621923681.pkl")
    print(sll['dTgen']['t_plot'][-1])
    print(sll.x_plot.shape)
    print(sll.u_plot.shape)
    print(sll.F_plot.shape)
    print(sll.t_plot.shape)
    print(sll.itp(0))
    print(sll.itp(100))