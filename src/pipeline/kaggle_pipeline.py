# -*- coding: utf-8 -*-
import features.preparator as preparator
import pandas as pd


class KagglePipeline:
    """ This class provides a pipeline to predict a test file from Kaggle.
    """

    def __init__(self, classifier, training_dataset, test_dataset):
        """ Constructor.

            Arg:
                classifier the classifier to use
        """
        self.classifier = classifier
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.header = 'id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,Acer_Opalus,Acer_Palmatum,Acer_Pictum,Acer_Platanoids,Acer_Rubrum,Acer_Rufinerve,Acer_Saccharinum,Alnus_Cordata,Alnus_Maximowiczii,Alnus_Rubra,Alnus_Sieboldiana,Alnus_Viridis,Arundinaria_Simonii,Betula_Austrosinensis,Betula_Pendula,Callicarpa_Bodinieri,Castanea_Sativa,Celtis_Koraiensis,Cercis_Siliquastrum,Cornus_Chinensis,Cornus_Controversa,Cornus_Macrophylla,Cotinus_Coggygria,Crataegus_Monogyna,Cytisus_Battandieri,Eucalyptus_Glaucescens,Eucalyptus_Neglecta,Eucalyptus_Urnigera,Fagus_Sylvatica,Ginkgo_Biloba,Ilex_Aquifolium,Ilex_Cornuta,Liquidambar_Styraciflua,Liriodendron_Tulipifera,Lithocarpus_Cleistocarpus,Lithocarpus_Edulis,Magnolia_Heptapeta,Magnolia_Salicifolia,Morus_Nigra,Olea_Europaea,Phildelphus,Populus_Adenopoda,Populus_Grandidentata,Populus_Nigra,Prunus_Avium,Prunus_X_Shmittii,Pterocarya_Stenoptera,Quercus_Afares,Quercus_Agrifolia,Quercus_Alnifolia,Quercus_Brantii,Quercus_Canariensis,Quercus_Castaneifolia,Quercus_Cerris,Quercus_Chrysolepis,Quercus_Coccifera,Quercus_Coccinea,Quercus_Crassifolia,Quercus_Crassipes,Quercus_Dolicholepis,Quercus_Ellipsoidalis,Quercus_Greggii,Quercus_Hartwissiana,Quercus_Ilex,Quercus_Imbricaria,Quercus_Infectoria_sub,Quercus_Kewensis,Quercus_Nigra,Quercus_Palustris,Quercus_Phellos,Quercus_Phillyraeoides,Quercus_Pontica,Quercus_Pubescens,Quercus_Pyrenaica,Quercus_Rhysophylla,Quercus_Rubra,Quercus_Semecarpifolia,Quercus_Shumardii,Quercus_Suber,Quercus_Texana,Quercus_Trojana,Quercus_Variabilis,Quercus_Vulcanica,Quercus_x_Hispanica,Quercus_x_Turneri,Rhododendron_x_Russellianum,Salix_Fragilis,Salix_Intergra,Sorbus_Aria,Tilia_Oliveri,Tilia_Platyphyllos,Tilia_Tomentosa,Ulmus_Bergmanniana,Viburnum_Tinus,Viburnum_x_Rhytidophylloides,Zelkova_Serrata'

    def predict(self, output_filename):
        """ Predicts the dataset and outputs the answers in the output_filename.

            Arg:
                output_filename the name of the file to write to
        """
        self.prepare_training_dataset()
        self.prepare_test_dataset()
        self.train_on_full_dataset()

        predictions = self.classifier.predict_proba(self.test_dataset.drop('id', axis=1))

        df = pd.DataFrame()

        columns = self.header.split(',')[1:]
        df['id'] = self.test_dataset['id']

        for k, column_name in enumerate(columns):
            df[column_name] = predictions[:,k]

        df.to_csv(output_filename, index=False)

    def prepare_test_dataset(self):
        """ Prepares the dataset. """
        pca = preparator.PCAPreparator()
        pca.prepare(self.test_dataset, self.test_dataset.columns[1:])

        standardizer = preparator.StandardizerPreparator()
        standardizer.prepare(self.test_dataset, self.test_dataset.columns[1:]) # We don't apply preparation on the IDs (=target) columns.

    def prepare_training_dataset(self):
        """ Prepares the dataset. """
        pca = preparator.PCAPreparator()
        pca.prepare(self.training_dataset, self.training_dataset.columns[1:])

        standardizer = preparator.StandardizerPreparator()
        standardizer.prepare(self.training_dataset, self.training_dataset.columns[1:]) # We don't apply preparation on the Species (=target) columns.

        name2int = preparator.Name2IntPreparator()
        name2int.prepare(self.training_dataset, ['species']) # Replace the species name by an integer.

    def train_on_full_dataset(self):
        """ Trains the classifier on the full dataset. """
        self.classifier.train(self.training_dataset.drop('species', axis=1), self.training_dataset['species'])
