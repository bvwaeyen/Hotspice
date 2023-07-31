# Created 19/07/23 to analyse the data created by clocking_KQ.py

import hotspice
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# data file stuff
def data_directory(lattice_angle):
    return f"KQ/angle {int(lattice_angle * 180. / np.pi)}/"
def data_filename(B, sample):
    return f"KQ B {B*1e3:.1f} mT, sample {sample}.json"

class ResultReader:
    def __init__(self, lattice_angle, outputreader_name="full", n_out=1):
        """<lattice_angle> is fixed. Use 'full' or 'regional' as outputreader names. n_out x n_out is used in case of 'regional'."""

        self.lattice_angle = lattice_angle
        self.mm = hotspice.ASI.IP_Pinwheel(a=248e-9, n=21, moment=3e-16, angle=lattice_angle)  # dummy ASI, DON'T USE FOR EXPERIMENTS
        datastream = hotspice.io.RandomBinaryDatastream()  # dummy datastream
        inputter = hotspice.io.ClockingFieldInputter(datastream)  # dummy inputter
        self.fulloutputreader = hotspice.io.FullOutputReader(self.mm)  # Full resolution for reconstruction

        if outputreader_name.lower().strip() == "full":
            self.outputreader = hotspice.io.FullOutputReader(self.mm)
        else:
            self.outputreader = hotspice.io.RegionalOutputReader(n_out, n_out, self.mm)

        self.experiment = hotspice.experiments.KernelQualityExperiment(inputter, self.outputreader, self.mm)

    def get_results(self, B, sample):
        """Loads json data from <B> and <sample> and reconstructs whole experiment with new outputreader.
        Returns [K, G, Q, k, g, q]"""

        raw_df = pd.read_json(data_directory(self.lattice_angle) + data_filename(B, sample))
        new_y = [self.outputreader.read_state(self.fulloutputreader.unread_state(raw_output, self.mm)).copy() for raw_output in raw_df["y"]]
        new_df = pd.DataFrame({"metric": raw_df["metric"], "inputs": raw_df["inputs"], "y": new_y})

        self.experiment.load_dataframe(new_df)
        self.experiment.calculate_all()
        results = self.experiment.results
        K, G, Q, k, g, q = results["K"], results["G"], results["K"] - results["G"], results["k"], results["g"], results["k"] - results["g"]
        return [K, G, Q, k, g, q]


# dummy ASI essential parameters
lattice_angles = np.array([0., 4., 8., 12.]) * np.pi/180.  # varies in {0°, 4°, 8°, 12°}

# magnetic field input
B_min, B_max, dB = 55, 63, 0.1  # in mT
B_array = np.arange(B_min, B_max + dB, dB) * 1e-3  # in T

# experiment
samples = 20  # also determines seed of E_B (and Néel and the KQ Experiments I suppose)

# ----------------------------------------------------------------------------------------------------
# Calculating metrics

def array_filename(name, n_out=1):
    array_filename = f"KQ/metrics/{name}"
    if name == "regional": array_filename += f" {n_out}"
    array_filename += ".npy"
    return array_filename

def human_name(name="full", n_out=1):
    return f"{f'{n_out}x{n_out} ' if name=='regional' else ''}{name} output"


recalculate = False
names = 10*["regional"] + ["full"]
n_outs = range(1, 12)
if recalculate:
    print("Calculating results")
    for name, n_out in zip(names, n_outs):  # full and 10 regionals
        print(human_name(name, n_out))

        if os.path.exists(array_filename(name, n_out)):
            print("File already exists! Skipping this one!")
            continue

        metrics = np.zeros((6, lattice_angles.size, B_array.size, samples))  # K G Q k g q

        for angle_i, lattice_angle in enumerate(lattice_angles):
            print(f"angle {int(lattice_angle * 180./np.pi)}°")
            resultreader = ResultReader(lattice_angle, outputreader_name=name, n_out=n_out)

            for B_i, B in enumerate(B_array):
                for sample in range(samples):
                    results = resultreader.get_results(B, sample)
                    metrics[:, angle_i, B_i, sample] = results

        np.save(array_filename(name, n_out), metrics)


# ----------------------------------------------------------------------------------------------------
# Hard numerical facts

full_metrics = np.load(array_filename("full"), "r")  # [metric_index, lattice_angle_index, B_index, sample]
full_means = np.mean(full_metrics, axis=-1)  # average out samples
for angle_i, lattice_angle in enumerate(lattice_angles):
    best_B_i = np.argmax(full_means[2, angle_i, :])
    best_Q = full_means[2, angle_i, best_B_i]
    print(f"The best Q = {best_Q} for {int(lattice_angle * 180./np.pi)}° at full resolution is at B = {B_array[best_B_i]*1e3:.1f}mT (index {best_B_i})")
print("")
regional_metrics = np.load(array_filename("regional", 5), "r")  # [metric_index, lattice_angle_index, B_index, sample]
regional_means = np.mean(regional_metrics, axis=-1)  # average out samples
for angle_i, lattice_angle in enumerate(lattice_angles):
    best_B_i = np.argmax(regional_means[2, angle_i, :])
    best_Q = regional_means[2, angle_i, best_B_i]
    print(f"The best Q = {best_Q} for {int(lattice_angle * 180./np.pi)}° at 5x5 resolution is at B = {B_array[best_B_i]*1e3:.1f}mT (index {best_B_i})")


# ----------------------------------------------------------------------------------------------------
# Plotting metrics
# use [metric_index, lattice_angle_index, B_index, sample] in metrics array

def metric_max(name="full", n_out=1):
    if name=="full":
        return 220
    return 2 * (n_out ** 2)

matplotlib.rcParams.update({'font.size': 12})  # larger font
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
interesting_names = ["full", "regional"]
interesting_n_outs = [0, 5]  # 6x6 is highest q for 8° with 5x5 very close second. 5x5 is more squinting, so can better illustrate other points

# -------------------------
"""
4 plots which clearly show average Q=K-G
fixed n_out -> KGQ
different angles have different axes, but metrics on the same plot
only shows averages
"""
show4 = False  # True
if show4:

    for name, n_out in zip(interesting_names, interesting_n_outs):
        if not os.path.exists(array_filename(name, n_out)):
            print(f"File '{array_filename(name, n_out)}' not found!")
            continue

        metrics = np.load(array_filename(name, n_out), "r")
        means = np.mean(metrics, axis=-1)

        fig4, axs4 = plt.subplots(2, 2, sharex="all", sharey="all")
        flat_axs4 = axs4.flatten()
        fig4.suptitle(f"Averages of {human_name(name, n_out)}")

        flat_axs4[0].set_xlim([57, 63])  # lower than 57 is never interesting
        flat_axs4[0].set_ylim([0, metric_max(name, n_out)])
        for ax in axs4[-1, :]: ax.set_xlabel("B (mT)")
        for ax in axs4[:, 0]: ax.set_ylabel("rank")

        for angle_i, lattice_angle in enumerate(lattice_angles):
            ax4 = flat_axs4[angle_i]
            ax4.set_title(f"{int(lattice_angle * 180./np.pi)}°")
            for metric_i, (metric_label, metric_marker) in enumerate(zip(["K", "G", "Q"], ["s", "D", "o"])):
                ax4.plot(B_array * 1e3, means[metric_i, angle_i, :], color=default_colors[metric_i], label="$\overline{"+metric_label+"}$")

        handles, labels = flat_axs4[0].get_legend_handles_labels()
        fig4.legend(handles, labels, loc="upper right")

        fig4.show()


# -------------------------
"""
3 plots which clearly show average Q=K-G
fixed n_out -> KGQ
different METRICS have different axes, but ANGLES on the same plot
only shows averages
"""
show3 = False
if show3:

    for name, n_out in zip(interesting_names, interesting_n_outs):
        if not os.path.exists(array_filename(name, n_out)):
            print(f"File '{array_filename(name, n_out)}' not found!")
            continue

        metrics = np.load(array_filename(name, n_out), "r")
        means = np.mean(metrics, axis=-1)

        fig3, axs3 = plt.subplots(1, 3, sharex="all", sharey="all")
        fig3.suptitle(f"Averages of {human_name(name, n_out)}")

        axs3[0].set_xlim([57, 63])  # lower than 57 is never interesting
        axs3[0].set_ylim([0, metric_max(name, n_out)])
        for ax in axs3[:]: ax.set_xlabel("B (mT)")
        axs3[0].set_ylabel("rank")

        for metric_i, metric_label in enumerate(["K", "G", "Q"]):
            ax3 = axs3[metric_i]
            ax3.set_title("$\overline{"+metric_label+"}$")
            for angle_i, (lattice_angle, ls) in enumerate(zip(lattice_angles, ["-", ":", "--", "-."])):
                ax3.plot(B_array * 1e3, means[metric_i, angle_i, :], color=default_colors[3+angle_i],
                         ls=ls, label=f"{int(lattice_angle * 180./np.pi)}°")

        handles, labels = axs3[-1].get_legend_handles_labels()
        fig3.legend(handles, labels, loc="center right")

        fig3.show()

# -------------------------
"""
12 plots which clearly show "errors"
fixed n_out -> KGQ
showing metrics and angles on different axes
showing quantiles like a box plot
"""
show12 = False  # True
if show12:

    for name, n_out in zip(interesting_names, interesting_n_outs):
        if not os.path.exists(array_filename(name, n_out)):
            print(f"File '{array_filename(name, n_out)}' not found!")
            continue

        metrics = np.load(array_filename(name, n_out), "r")
        medians = np.median(metrics, axis=-1)
        mins, maxs = np.min(metrics, axis=-1), np.max(metrics, axis=-1)
        first_quartiles, third_quartiles = np.quantile(metrics, 0.25, axis=-1), np.quantile(metrics, 0.75, axis=-1)

        fig12, axs12 = plt.subplots(lattice_angles.size, 3, sharex="all", sharey="all", figsize=(6.5, 8))
        fig12.suptitle(human_name(name, n_out))

        axs12[0, 0].set_xlim([57, 63])  # lower than 57 is never interesting
        axs12[0, 0].set_ylim([0, metric_max(name, n_out)])
        for ax in axs12[-1, :]: ax.set_xlabel("B (mT)")
        for ax in axs12[:, 0]: ax.set_ylabel("rank")

        for angle_i, lattice_angle in enumerate(lattice_angles):
            for metric_i, (metric_label, metric_marker) in enumerate(zip(["K", "G", "Q"], ["s", "D", "o"])):
                ax12 = axs12[angle_i, metric_i]
                ax12.plot(B_array * 1e3, medians[metric_i, angle_i, :], color=default_colors[metric_i], label="median")
                ax12.fill_between(B_array * 1e3, first_quartiles[metric_i, angle_i, :], third_quartiles[metric_i, angle_i, :],
                                  color=default_colors[metric_i], alpha=0.50, label="quartiles")
                ax12.fill_between(B_array * 1e3, mins[metric_i, angle_i, :], maxs[metric_i, angle_i, :],
                                  color=default_colors[metric_i], alpha=0.25, label="extrema")

        # 12 general titles
        for angle_i, lattice_angle in enumerate(lattice_angles):
            [[xmin, ymin], [xmax, ymax]] = axs12[angle_i, 0].get_position()._points
            fig12.text(0.02, 0.5*(ymax+ymin), f"{int(lattice_angle * 180./np.pi)}°", ha="center")
        for metric_i, metric_label in enumerate(["K", "G", "Q"]):
            [[xmin, ymin], [xmax, ymax]] = axs12[0, metric_i].get_position()._points
            fig12.text(0.5*(xmax+xmin), 0.92, metric_label, va="center")

        handles = [plt.Line2D([0], [0], label="median", color="grey"),
                   plt.fill_between([0], [0], label="quartiles", color="grey", alpha=0.5),
                   plt.fill_between([0], [0], label="extrema", color="grey", alpha=0.25)]
        [[xmin, ymin], [xmax, ymax]] = axs12[0, 1].get_position()._points
        fig12.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5*(xmax+xmin), ymax))

        fig12.show()


# -------------------------
"""
Clearly compares averaged metrics as a function of angles for fixed n_out
fixed n_out -> KGQ
maxed over field B, NOT FIXED, as this would be tweaked
"""
show_a = False  # True
if show_a:
    for name, n_out in zip(interesting_names, interesting_n_outs):
        if not os.path.exists(array_filename(name, n_out)):
            print(f"File '{array_filename(name, n_out)}' not found!")
            continue

        metrics = np.load(array_filename(name, n_out), "r")
        means = np.mean(metrics, axis=-1)  # averaging over samples
        best_Bs = np.argmax(means[2, :, :], axis=-1)  # B for best Q for every angle

        fig_a, ax_a= plt.subplots(figsize=(5.2, 4))  # a from angle
        ax_a.set_title(human_name(name, n_out))
        ax_a.set_ylim([0, metric_max(name, n_out)])
        ax_a.set_xlabel(r"$\alpha (^\circ)$")
        ax_a.set_ylabel("best rank")

        for metric_i, (metric_label, marker) in enumerate(zip(["K", "G", "Q"], ["s", "D", "o"])):
            y = [means[metric_i, angle_i, best_Bs[angle_i]] for angle_i in range(lattice_angles.size)]
            ax_a.plot(np.int16(lattice_angles * 180./np.pi), y, color=default_colors[metric_i], marker=marker,
                      label=r"$\overline{"+metric_label+r"}_{best}$")

        ax_a.legend()
        fig_a.show()


# -------------------------
"""
Clearly varying n_out for different metrics and angles
"""
show_n4 = False
if show_n4:
    for Q_or_q in ["q", "Q"]:
        fig_n4, axs_n4 = plt.subplots(2, 2, sharex="all", sharey="all")
        flat_axs_n4 = axs_n4.flatten()

        flat_axs_n4[0].set_xlim([2, 220])
        flat_axs_n4[0].set_ylim([0, 220 if Q_or_q=="Q" else 1])

        for ax in axs_n4[-1, :]: ax.set_xlabel("n")
        for ax in axs_n4[:, 0]: ax.set_ylabel("best rank" if Q_or_q=="Q" else "best normalized rank")

        ns = [metric_max(name, n_out) for (name, n_out) in zip(names, n_outs)]
        n_mean_metrics = np.zeros((len(n_outs), 6, lattice_angles.size))
        for n_out_i, (name, n_out) in enumerate(zip(names, n_outs)):
            metrics = np.load(array_filename(name, n_out), "r")
            means = np.mean(metrics, axis=-1)  #average out samples
            best_Bs = np.argmax(means[2, :, :], axis=-1)  # B for best Q for every angle
            for angle_i in range(lattice_angles.size):  # assign every metric and every angle the best B according to Q
                n_mean_metrics[n_out_i, :, angle_i] = means[:, angle_i, best_Bs[angle_i]]

        for angle_i, lattice_angle in enumerate(lattice_angles):
            ax = flat_axs_n4[angle_i]
            ax.set_title(f"{int(lattice_angle * 180./np.pi)}°")
            for metric_i, (metric_label, metric_marker) in enumerate(zip((["K", "G", "Q"] if Q_or_q=="Q" else ["k", "g", "q"]), ["s", "D", "o"])):
                ax.plot(ns, n_mean_metrics[:, metric_i if Q_or_q=="Q" else metric_i+3, angle_i], color=default_colors[metric_i],
                        label=r"$\overline{"+metric_label+r"}_{\rm best}$", marker=metric_marker)

        flat_axs_n4[0].set_xticks(ns[::2])
        handles, labels = axs_n4[0,0].get_legend_handles_labels()
        fig_n4.legend(handles, labels, loc="upper right")
        fig_n4.show()

# -------------------------
"""
Clearly varying n_out for Q and q for different angles
"""
show_na = True  # True
if show_na:
    for Q_or_q in ["q", "Q"]:
        fig_na, ax_na = plt.subplots(figsize=(5.5, 4.2))

        ax_na.set_xlim([2, 220])
        ax_na.set_ylim([0, 220 if Q_or_q=="Q" else 1])

        ax_na.set_xlabel("n")
        ax_na.set_ylabel(r"$\overline{"+Q_or_q+r"}_{best}$")

        ns = [metric_max(name, n_out) for (name, n_out) in zip(names, n_outs)]
        n_mean_q = np.zeros((len(n_outs), lattice_angles.size))
        for n_out_i, (name, n_out) in enumerate(zip(names, n_outs)):
            metrics = np.load(array_filename(name, n_out), "r")
            means = np.mean(metrics, axis=-1)  #average out samples
            best_Bs = np.argmax(means[2, :, :], axis=-1)  # B for best Q for every angle
            for angle_i in range(lattice_angles.size):  # assign every metric and every angle the best B according to Q
                n_mean_q[n_out_i, angle_i] = means[2 if Q_or_q=="Q" else 5, angle_i, best_Bs[angle_i]]

        for angle_i, (lattice_angle, ls) in enumerate(zip(lattice_angles, ["-", ":", "--", "-."])):
            ax_na.plot(ns, n_mean_q[:, angle_i], color=default_colors[3+angle_i],
                       label=f"{int(lattice_angle * 180./np.pi)}°", marker="o", ls=ls)

        ax_na.set_xticks(ns[::])
        ax_na.legend()
        fig_na.show()

# -------------------------
# -------------------------
"""
Showing evolution of B distribution of normalized ranks for different n_out
"""
show_Bn = False  # True
if show_Bn:
    for Q_or_q in ["q", "Q"]:
        fig_Bn, axs_Bn = plt.subplots(4, 3, sharex="all", sharey="all", figsize=(6.5, 8))
        flat_axs_Bn = axs_Bn.flatten()

        for ax in axs_Bn[-1, :]: ax.set_xlabel("n")
        for ax in axs_Bn[:, 0]: ax.set_ylabel("B (mT)")
        flat_axs_Bn[0].set_ylim([57, 63])

        ns = np.array([metric_max(name, n_out) for (name, n_out) in zip(names, n_outs)])
        Ns, Bs = np.meshgrid(ns, B_array * 1e3)  # mT
        Bn_mean_metrics = np.zeros((6, lattice_angles.size, B_array.size, ns.size,))  # Q, a, B, n
        for n_out_i, (name, n_out) in enumerate(zip(names, n_outs)):
            metrics = np.load(array_filename(name, n_out), "r")  # [metric_index, lattice_angle_index, B_index, sample] in metrics array
            means = np.mean(metrics, axis=-1)  # average out samples
            Bn_mean_metrics[:, :, :, n_out_i] = means[:, :, :]

        for angle_i, lattice_angle in enumerate(lattice_angles):
            for metric_i, (metric_label, metric_marker) in enumerate(zip((["K", "G", "Q"] if Q_or_q=="Q" else ["k", "g", "q"]), ["s", "D", "o"])):
                ax = axs_Bn[angle_i, metric_i]
                im = ax.pcolormesh(Ns, Bs, Bn_mean_metrics[metric_i if Q_or_q=="Q" else metric_i+3, angle_i, :, :],
                                   shading="nearest", vmin=0, vmax=220 if Q_or_q == "Q" else 1)

        # 12 general titles
        for angle_i, lattice_angle in enumerate(lattice_angles):
            [[xmin, ymin], [xmax, ymax]] = axs_Bn[angle_i, 0].get_position()._points
            fig_Bn.text(0.02, 0.5*(ymax+ymin), f"{int(lattice_angle * 180./np.pi)}°", ha="center")
        for metric_i, metric_label in enumerate((["K", "G", "Q"] if Q_or_q=="Q" else ["k", "g", "q"])):
            [[xmin, ymin], [xmax, ymax]] = axs_Bn[0, metric_i].get_position()._points
            fig_Bn.text(0.5*(xmax+xmin), 0.90, r"$\overline{"+metric_label+r"}$", va="center")

        flat_axs_Bn[0].set_xticks(ns[::2])
        cbar_ax = fig_Bn.add_axes([0.92, 0.15, 0.02, 0.7])
        fig_Bn.colorbar(im, cax=cbar_ax)
        fig_Bn.show()

plt.show()


