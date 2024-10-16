import umap.umap_ as umap
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import plotnine as p9
import plotly.express as px

import numpy as np
import seaborn as sns
import pandas as pd
# Fix numpy seed
# Fix torch seed

class Visualize:
    def __init__(self, seed=42):
        self.seed = seed

    def encode_cat(self, dFrame):
        """
        Transform categorical columns into a binary expansion
        :param dFrame: the dataframe to encode
        :return: The encoded version
        """

        cat_clm = dFrame.select_dtypes(exclude=["int", 'float', 'double']).columns.tolist()
        return pd.get_dummies(dFrame, columns=cat_clm)

    def umap(self, dFrame, min_dist=0.2, n_components=2, plots=False):
        """
        Visualization with umap
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        umap_data = umap.UMAP(min_dist=min_dist, n_components=n_components).fit_transform(dFrame.values)
        if plots:
            self.plots("Decomposition using UMAP", umap_data, n_components)
        return umap_data

    def tsne(self, dFrame, n_components=2, plots=False):
        """
        Visualisation with t-sne
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        tsne = TSNE(n_components=n_components).fit_transform(dFrame.values)
        if plots:
            self.plots("t-SNE components", tsne, n_components)
        return tsne

    def pca(self, dFrame, n_components=2, plots=False):
        """
        Visualisation with pca
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(dFrame.values)
        plt.figure(figsize=(12, 8))
        if plots:
            self.plots("Component-wise and Cumulative Explained Variance", pca_result, n_components)
        return pca_result

    def svd(self, dFrame, n_components=2, plots=False):
        """
        Single value decomposition visualisation
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        svd = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(dFrame.values)
        if plots:
            self.plots("SVD Components", svd, n_components)
        return svd

    def plots(self, title, components, n_components=2):
        plt.figure(figsize=(12, 8))
        plt.title(title)
        if n_components > 2:
            for i in range(n_components):
                sns.scatterplot(components[:, i % n_components], components[:, (i+1) % n_components])
        else:
            sns.scatterplot(components[:, 0], components[:, 1])

    def reachability_plot(self, labels, reachability, indexes, savefig=None):
        """
        Plot the reachability of all clusters
        :param labels: the clusters labels
        :param reachability: the indexes reachability
        :param indexes: the dataset indexes
        :return: Nothing
        """
        df = pd.DataFrame.from_dict({"reachability": reachability, "indexes": indexes, "labels": labels})
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df, x="indexes", y="reachability", hue="labels")

        if savefig is not None:
            plt.savefig(savefig)
            fig = px.scatter(df, x='indexes', y='reachability', color='labels')
            fig.update_layout(height=600, width=1000, )
            fig.write_html("{}-px.html".format(savefig))

    def clusters_plots(self, dFrame, labels, dimRedFn="umap", savefig=None, *args, **kwargs):
        """
        Shows the clusters using dimensionality reduction
        :param dFrame: the original dataFrame (n_samples, n_features)
        :param labels: the clusters label (n_samples)
        :param dimRedFn: dimensionality reduction technique, options are: "umap", "tsne", "pca", "svd"
        should return 2 components
        :param args: dimensionality reduction arguments
        :param kwargs: dimensionality reduction keyword arguments
        """
        if "umap" in dimRedFn.lower():
            components = self.umap(dFrame, *args, **kwargs)
        elif "tsne" in dimRedFn.lower():
            components = self.tsne(dFrame, *args, **kwargs)
        elif "pca" in dimRedFn.lower():
            components = self.pca(dFrame, *args, **kwargs)
        elif "svd" in dimRedFn.lower():
            components = self.svd(dFrame, *args, **kwargs)
        else:
            components = self.dimRedFn(self.encode_cat(dFrame), *args, **kwargs)

        df = pd.DataFrame.from_dict({
            "labels": labels.squeeze(),
            "1st_component": components[:, 0],
            "2nd_component": components[:, 1],
        })
        df["labels"] = df["labels"].astype(str)
        plt.figure()
        sns.scatterplot(data=df, x="1st_component", y="2nd_component", hue="labels")
        if savefig is not None:
            # plt.savefig(savefig)
            plot = p9.ggplot(data=df, mapping=p9.aes(x='1st_component', y='2nd_component', color='labels'))
            plot += p9.geom_point()
            plot.save(savefig, width=10, height=10, dpi=300)
            fig = px.scatter(df, x='1st_component', y='2nd_component', color='labels')
            fig.update_layout(height=600, width=1000,)
            fig.write_html("{}-px.html".format(savefig))
