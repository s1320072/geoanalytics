.. geoanalytics documentation master file, created by
   sphinx-quickstart on Tue Jul 22 15:12:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to geoanalytics's documentation!
========================================

geoAnalytics is an open-source Python-based Machine Learning library developed to discover various forms of useful information hidden in the raster data. The algorithms provided in this library cover a wide-spectrum of machine learning tasks, such as imputation, image fusion, clustering, classification, one class classification, and pattern mining. This library being platform independent can run any operating system.

The library has been designed with a strong research focus, offering robust tools tailored to the needs of geospatial, remote sensing, and planetary data analysis. Its rich collection of modules enables users to process, analyze, and visualize complex spatial datasets with ease.


Areas of Research
-----------------

Geospatial & Spatiotemporal Data Analysis and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This area focuses on analyzing spatial data from sources like satellite imagery, remote sensing, and georeferenced measurements. The library data in supports raster, vector and tabular formats such as GeoTIFF, NetCDF, and CSV. It includes tools for preprocessing, such as handling missing values, extracting spatial coordinates, and converting raster to tabular form. Users can apply clustering, classification, imputation and pattern mining methods for geospatial contexts. Clustering algorithms like k-Means++, DBSCAN, BIRCH, Spectral Clustering, and Mean Shift are used to group data based on spatial, spectral, or temporal similarities. These are useful for land cover classification, urban development tracking, and anomaly detection.

Spatiotemporal mining techniques discover patterns that recur over time and space. These are applied in climate monitoring and planetary surface studies. Advanced methods also support the identification of periodic and partially periodic spatial patterns. Visualization features such as plotting clusters, nearest points, and spatial boundaries support exploratory analysis and interpretation. The library also integrates with PostgreSQL/PostGIS, allowing users to insert, update, or convert raster data directly from databases.

Data Imputation and Applications in Remote Sensing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Incomplete data is common in remote sensing due to sensor errors or obstructions. This area addresses such issues using various imputation techniques. Basic methods include forward/backwards fill and statistical approaches like mean, median, and mode. Advanced methods include KNN, MICE, and Hot Deck imputation.

Imputed data is then ready for classification, clustering, and pattern identification, ensuring higher reliability in remote sensing applications.

Planetary and Astrophysical Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library also supports analysis of planetary and astrophysical imagery. It has been applied to lunar satellite data for clustering and imputation. These tools can be extended to other datasets, such as Mars rover images or asteroid observations, making them valuable in space research and planetary science.

To explore the available modules and dive into the API documentation, refer to the navigation below:

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
