/**
 * This program is used by <i>Hierarchical Sampling for Multi-Instance Ensemble Learning</i>.
 * 
 */
package moe.test;

import java.io.File;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * Class for Cross-Validation.
 * 
 * @author Meng Fang (Meng.Fang@student.uts.edu.au)
 * @version $Revision: 2.0 $
 */
public class CrossValidation {

	private int iFolds;

	private Instances dataset;
	private Instances[] trainset;
	private Instances[] testset;
	private int numInstances;
	private int classIndex;
	private Instance instance;
	private String fileName;

	public CrossValidation(String fileName_in, int ifolds) throws Exception {
		fileName = fileName_in;
		dataset = getArffData(fileName_in);
		Random r = new java.util.Random();
		dataset.randomize(r);
		// classIndex = dataset_in.numAttributes()-1;
		numInstances = dataset.numInstances();
		iFolds = ifolds;
		trainset = new Instances[iFolds];
		testset = new Instances[iFolds];
		classIndex = dataset.numAttributes() - 1;

		instanceDispatch();
	}

	public CrossValidation(Instances dbIn, int ifolds) throws Exception {
		dataset = new Instances(dbIn);
		Random r = new java.util.Random();
		dataset.randomize(r);
		classIndex = dbIn.numAttributes() - 1;
		numInstances = dataset.numInstances();
		iFolds = ifolds;
		trainset = new Instances[iFolds];
		testset = new Instances[iFolds];

		instanceDispatch();
	}

	public Instances getRandomizedSet() {
		return (dataset);
	}

	public Instances getTrainSubset(int isubset) {
		return (trainset[isubset]);
	}

	public Instances getTestSubset(int isubset) {
		return (testset[isubset]);
	}

	public Instances getArffData(String fileName) throws Exception {
		File inputFile = new File(fileName);
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		return atf.getDataSet();
	}

	private double instanceDispatch() throws Exception {

		Random r = new java.util.Random();
		dataset.randomize(r);
		int average = numInstances / iFolds;
		int last = numInstances - (iFolds - 1) * average;
		int i, j, k;
		for (i = 0; i < (iFolds - 1); i++) {
			testset[i] = new Instances(dataset);
			testset[i].setClassIndex(classIndex);
			testset[i].delete();
			trainset[i] = new Instances(dataset);
			trainset[i].setClassIndex(classIndex);
			trainset[i].delete();

			for (j = 0; j < average; j++) {
				instance = dataset.instance(j + i * average);
				testset[i].add(instance);
			}
		}
		testset[iFolds - 1] = new Instances(dataset);
		testset[iFolds - 1].setClassIndex(classIndex);
		testset[iFolds - 1].delete();
		trainset[iFolds - 1] = new Instances(dataset);
		trainset[iFolds - 1].setClassIndex(classIndex);
		trainset[iFolds - 1].delete();

		for (i = 0; i < last; i++) {
			instance = dataset.instance(numInstances - 1 - i);
			testset[iFolds - 1].add(instance);
		}

		for (i = 0; i < iFolds; i++) {
			for (j = 0; j < testset[i].numInstances(); j++) {
				instance = testset[i].instance(j);
				for (k = 0; k < iFolds; k++) {
					if (k != i) {
						trainset[k].add(instance);
					}
				}
			}

		}
		return 0;
	}
}
