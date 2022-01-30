import psycopg2
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw

def connect2DB(user = 'chembl', password='chembl', host='127.0.0.1', port='5432', database='chembl_25'):
    connection = psycopg2.connect(user = user,
                    password = password,
                    host = host,
                    port = port,
                    database = database)
    return connection

def getSmilesWithActivity(connection, nRows = None):
    cursor = connection.cursor()
    columns = ['molregno', 'canonical_smiles', 'activity_id', 
           'standard_value', 'standard_units', 'standard_flag', 'standard_type', 'activity_comment',
           'alogp', 'hba', 'hbd', 'psa', 'rtb', 'ro3_pass', 'num_ros_violations', 'molecular_species', 'full_mwt', 'aromatic_rings', 'heavy_atoms', 'qed_weighted']

    cursor.execute("select CS.molregno, \
               CS.canonical_smiles, \
               AC.activity_id, \
               AC.standard_value, \
               AC.standard_units, \
               AC.standard_flag, \
               AC.standard_type, \
               AC.activity_comment, \
               CP.ALOGP, \
               CP.HBA, \
               CP.HBD, \
               CP.PSA, \
               CP.RTB, \
               CP.RO3_PASS, \
               CP.NUM_RO5_VIOLATIONS, \
               CP.MOLECULAR_SPECIES, \
               CP.FULL_MWT, \
               CP.AROMATIC_RINGS, \
               CP.HEAVY_ATOMS, \
               CP.QED_WEIGHTED \
               from COMPOUND_STRUCTURES CS \
               inner join ACTIVITIES AC on CS.molregno = AC.molregno \
               inner join COMPOUND_PROPERTIES CP on CS.molregno = CP.MOLREGNO \
               and (AC.standard_type = 'IC50' or AC.standard_type = 'GI50' or AC.standard_type = 'Potency') \
               and (AC.standard_value IS NOT NULL)")
    if nRows is None:
        molData = pd.DataFrame(cursor.fetchall(), columns = columns)
    elif nRows > 0:
        molData = pd.DataFrame(cursor.fetchmany(nRows), columns = columns)
    return molData

def provideMoleculeStatistics(smiles):
    #print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    
    newSmiles = Chem.MolToSmiles(mol, canonical = True, isomericSmiles = False)
    negativeCharged = sum([ item.GetFormalCharge() if item.GetFormalCharge() < 0 else 0 for item in mol.GetAtoms() ])
    positiveCharged = sum([ item.GetFormalCharge() if item.GetFormalCharge() > 0 else 0 for item in mol.GetAtoms() ])
    
    elementsList = list(set([atom.GetSymbol() for atom in mol.GetAtoms()]))
    numberOfRings = mol.GetRingInfo().NumRings()
    
    return(newSmiles, negativeCharged, positiveCharged, elementsList, numberOfRings)

import codecs
encodeToUTF8 = False
def canonicalizeSmilesAndProvideDescriptor(smiles):
    try:
        newSmiles, negativeCharged, positiveCharged, elementsList, numberOfRings = provideMoleculeStatistics(smiles)     
    except:
        newSmiles, negativeCharged, positiveCharged, elementsList, numberOfRings = (None, None, None, None, None)
        print('Exception!!! :', smiles)
        
    if (encodeToUTF8):
        return((codecs.encode(newSmiles, 'utf-8'), negativeCharged, positiveCharged, elementsList, numberOfRings))
    else:
        return((newSmiles, negativeCharged, positiveCharged, elementsList, numberOfRings))

if __name__ == '__main__':
    os.chdir('/home/rafalb/work/molecules/chemicalSmilesSpace/src')
    picklesDir = 'pickles/'
    DBread = False
    if DBread:
        connection = connect2DB(port='16001', database='chembl_28')
        molData = getSmilesWithActivity(connection)
        connection.close()
        molData.to_pickle(picklesDir+'molDataRaw.pkl')

    dataAggregate = False
    if dataAggregate:
        molData = pd.read_pickle(picklesDir+'molDataRaw.pkl')
        # what units do we have?
        unitAgg = molData.groupby('standard_units').agg('count').loc[:, 'molregno'].T
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(unitAgg)
        # take only the entries expressed in nM
        molData = molData[molData['standard_units']=='nM'].reset_index()
        
        aggFunctions = {
                    'molregno': ['min', 'count'], 'canonical_smiles': 'min',
                    'standard_value': ['min', 'max'],
                    'standard_type': 'min',
                    'alogp': ['min', 'max'],
                    'hba': ['min', 'max'],
                    'hbd': ['min', 'max'],
                    'psa': ['min', 'max'],
                    'rtb': ['min', 'max'],
                    'ro3_pass': 'min',
                    'num_ros_violations': 'min',
                    'molecular_species': 'min',
                    'full_mwt': ['min', 'max'],
                    'aromatic_rings': 'min',
                    'heavy_atoms': 'min',
                    'qed_weighted': ['min', 'max']
                    }
        grouped = molData.groupby('molregno')
        print('Aggregating...')
        molData = grouped.agg(aggFunctions).reset_index()
        molData.to_pickle(picklesDir+'molDataGrouped.pkl')

    provideDescriptors = False
    if provideDescriptors:
        molData = pd.read_pickle(picklesDir+'molDataGrouped.pkl')
        sourceColumn = ('canonical_smiles', 'min')
        nTotal = len(molData)
        nStart = 0
        nSize = 100000
        nBatch = np.ceil((nTotal - nStart)/nSize).astype(int)
        for iii in range(nBatch):
            iBeg = nStart + iii * nSize
            if (iii == nBatch - 1):
                iEnd = nTotal
            else:
                iEnd = nStart + (iii + 1) * nSize
            print('Batch ID: '+str(iii))
            result = molData.loc[iBeg:iEnd, sourceColumn].apply(canonicalizeSmilesAndProvideDescriptor)
            molData.loc[iBeg:iEnd, 'canonicalSmiles'] = [item[0] for item in result]
            molData.loc[iBeg:iEnd, 'negativeCharged'] = [item[1] for item in result]
            molData.loc[iBeg:iEnd, 'positiveCharged'] = [item[2] for item in result]
            molData.loc[iBeg:iEnd, 'elementsSet'] = [item[3] for item in result]
            molData.loc[iBeg:iEnd, 'numberOfRings'] = [item[4] for item in result]
        
        #organicChemistrySet = set(['B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'])
        #molData['organicChemistryElementsOnly'] = molData['elementsSet'].apply(lambda x: set(x) < organicChemistrySet)

        molData.to_pickle(picklesDir+'molDataDescriptors.pkl')

 

    cleanData = True
    if cleanData:     
        molData = pd.read_pickle(picklesDir+'molDataDescriptors.pkl') 
        
        molData = molData.dropna()
        molData = molData[~pd.isna(molData['elementsSet'])]
        print(molData.columns)
        organicChemistrySet = set(['B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'])

        molData['organicChemistryElementsOnly'] = molData['elementsSet'].apply(lambda x: set(x) < organicChemistrySet)

        staticDynamicFeatures = pd.DataFrame()
        toBeAveraged = ['standard_value', 'alogp', 'hba', 'hbd', 'psa', 'rtb', 'full_mwt', 'qed_weighted']
        for quantity in toBeAveraged:
            staticDynamicFeatures.loc[:, quantity] = (molData.loc[:, (quantity, 'min')] + molData.loc[:, (quantity, 'max')])/2
            staticDynamicFeatures.loc[:, quantity].astype(float)

        toBeTaken = ['aromatic_rings', 'heavy_atoms']
        for quantity in toBeTaken:
            staticDynamicFeatures.loc[:, quantity] = molData.loc[:, (quantity, 'min')]
            staticDynamicFeatures.loc[:, quantity].astype(float)

        staticDynamicFeatures.loc[:, 'number_of_rings'] = molData.loc[:, 'numberOfRings'].astype(float)
        staticDynamicFeatures.loc[:, 'standard_type'] = molData.loc[:, ('standard_type', 'min')]

        toBeTaken = ['canonicalSmiles', 'negativeCharged', 'positiveCharged', 'elementsSet', 'numberOfRings', 'organicChemistryElementsOnly']
        for quantity in toBeTaken:
            staticDynamicFeatures[quantity] = molData.loc[:, quantity]
        print(molData.head())
        print(molData.columns)
        print(staticDynamicFeatures.head())
        print(staticDynamicFeatures.columns)
    pass
    