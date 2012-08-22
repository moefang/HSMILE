/**
 * This program is used by <i>Hierarchical Sampling for Multi-Instance Ensemble Learning</i>.
 * 
 */
package moe.sampling;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Debug.Random;

/**
 * Class for different sampling strategies for Multiple Instances Inter- and
 * Intra- Sampling.
 * 
 * @author Meng Fang (Meng.Fang@student.uts.edu.au)
 * @version $Revision: 2.0 $
 * 
 */
public class MISamplingStrategy {

	Instances bags = null;

	Instances negativeBags;

	double miu = 4.5;

	public MISamplingStrategy(Instances initialData, double initialMiu) {
		this.bags = initialData;
		int numBags = bags.numInstances();

		this.miu = initialMiu;

		negativeBags = new Instances(initialData, 0);

		for (int i = 0; i < numBags; i++) {
			Instance current = bags.instance(i);
			double classlabel = current.classValue();
			if (classlabel == 1.0) { // positive
				// do nothing
			} else { // negative
				negativeBags.add(initialData.instance(i));
			}
		}
	}

	public Instances DualSampling(Instances data) {
		Instances original = data;
		Instances dataafterinter = InterSampling(original);
		Instances dataafterintra = IntraSampling(dataafterinter);

		return dataafterintra;
	}

	public Instances DualSamplingWithNegativeOnly(Instances data) {
		Instances original = data;
		Instances dataafterinter = InterSampling(original);
		Instances dataafterintra = IntraSamplingWithNegativeOnly(dataafterinter);

		return dataafterintra;
	}

	public Instances DualSamplingWithNaiveTotalSampling(Instances data) {
		Instances original = data;
		Instances dataafterinter = InterSampling(original);
		Instances dataafterintra = IntraSamplingByRandom(dataafterinter);

		return dataafterintra;
	}

	private Instances IntraSamplingByRandom(Instances data) {
		Instances original = data;
		int numData = original.numInstances();
		Instances newdata = new Instances(original, 0);

		for (int i = 0; i < numData; i++) {
			Instance current = original.instance(i);
			double classlabel = current.classValue();
			if (classlabel == 1.0) { // positive
				Instance newInst = positiveBagSamplingByRandom(current);
				newdata.add(current);

			} else { // negative
				Instance newInst = negativeBagSampling(current);
				newdata.add(newInst);
			}
		}

		return newdata;
	}

	private Instance positiveBagSamplingByRandom(Instance inst) {
		Instance bag = inst;
		// String attrStr = bag.stringValue(1);
		// String[] samplesStr = attrStr.split("\\n");
		// int len = samplesStr.length;
		int len = inst.attribute(1).relation().numAttributes();

		Random rd = new Random();

		Instance newInst = new Instance(inst.numAttributes());
		newInst.setDataset(inst.dataset());

		rd.setSeed(System.currentTimeMillis());

		Instances aftersampling = inst.attribute(1).relation()
				.stringFreeStructure();

		for (int j = 0; j < len; j++) {
			aftersampling
					.add(inst.relationalValue(1).instance(rd.nextInt(len)));
		}
		int relationValue;
		relationValue = newInst.attribute(1).addRelation(aftersampling);
		newInst.setValue(0, inst.value(0));
		newInst.setValue(1, relationValue);
		newInst.setValue(2, inst.classValue());

		return newInst;
	}

	private Instances InterSampling(Instances data) {
		Instances original = data;
		Random rd = new Random();
		rd.setSeed(System.currentTimeMillis());
		Instances newdata = original.resample(rd);
		return newdata;
	}

	private Instances IntraSampling(Instances data) {
		Instances original = data;
		int numData = original.numInstances();
		Instances newdata = new Instances(original, 0);

		for (int i = 0; i < numData; i++) {
			Instance current = original.instance(i);
			double classlabel = current.classValue();
			if (classlabel == 1.0) { // positive
				Instance newInst = positiveBagSampling(current);
				newdata.add(newInst);
			} else {
				Instance newInst = negativeBagSampling(current);
				newdata.add(newInst);
			}
		}
		return newdata;
	}

	private Instances IntraSamplingWithNegativeOnly(Instances data) {
		Instances original = data;
		int numData = original.numInstances();
		Instances newdata = new Instances(original, 0);

		for (int i = 0; i < numData; i++) {
			Instance current = original.instance(i);
			double classlabel = current.classValue();
			if (classlabel == 1.0) { // positive
				newdata.add(current);

			} else { // negative
				Instance newInst = negativeBagSampling(current);
				newdata.add(newInst);
			}
		}

		return newdata;
	}

	private Instance positiveBagSampling(Instance inst) {
		Instance bag = inst;
		String attrStr = bag.stringValue(1);
		String[] samplesStr = attrStr.split("\\n");
		int len = samplesStr.length;
//		int len = inst.attribute(1).relation().numAttributes();
		double[] postiveProbs = new double[len];
		double sumprobs = 0.0;
		double maxProb = 0.0;
		int maxIndex = 0;
		for (int i = 0; i < len; i++) {
			String sampleStr = samplesStr[i];
			String[] features = sampleStr.split(",");
			double[] instVal = new double[features.length];
			for (int k = 0; k < features.length; k++) {
				instVal[k] = Double.parseDouble(features[k]);
			}

			postiveProbs[i] = getPositiveProb(instVal);
			sumprobs += postiveProbs[i];

			if (postiveProbs[i] > maxProb) {
				maxProb = postiveProbs[i];
				maxIndex = i;
			}
		}

		Random rd = new Random();

		Instance newInst = new Instance(inst.numAttributes());
		newInst.setDataset(inst.dataset());

		rd.setSeed(System.currentTimeMillis());

		Instances aftersampling = inst.attribute(1).relation()
				.stringFreeStructure();

		for (int j = 0; j < len - 1; j++) {
			int samplingIndex = rd.nextInt(len);

			while (samplingIndex == maxIndex)
				samplingIndex = rd.nextInt(len);

			aftersampling.add(inst.relationalValue(1).instance(samplingIndex));
		}
		aftersampling.add(inst.relationalValue(1).instance(maxIndex));

		int relationValue;
		relationValue = newInst.attribute(1).addRelation(aftersampling);
		newInst.setValue(0, inst.value(0));
		newInst.setValue(1, relationValue);
		newInst.setValue(2, inst.classValue());

		return newInst;
	}

	private Instances positiveBagSampling(Instance inst, int times) {
		Instance bag = inst;
		Instances newdata = new Instances(inst.dataset(), 0);

		Attribute attr = bag.attribute(2);
		String attrStr = attr.toString();
		String[] samplesStr = attrStr.split("\\n");
		int len = samplesStr.length;
		double[] postiveProbs = new double[len];
		double sumprobs= 0.0;
		double maxProb = 0.0;
		int maxIndex = 0;
		for (int i = 0; i < len; i++) {
			String sampleStr = samplesStr[i];
			String[] features = sampleStr.split(",");
			double[] instVal = new double[features.length];
			for (int k = 0; k < features.length; k++) {
				instVal[k] = Double.parseDouble(features[k]);
			}

			postiveProbs[i] = getPositiveProb(instVal);
			sumprobs += postiveProbs[i];

			if (postiveProbs[i] > maxProb) {
				maxProb = postiveProbs[i];
				maxIndex = i;
			}
		}

		Random rd = new Random();
		for (int i = 0; i < times; i++) {
			// Instance newInst = new Instance(bag);
			Instance newInst = bag;
			rd.setSeed(System.currentTimeMillis());
			StringBuffer sb = new StringBuffer();
			int j = 0;
			for (; j < len - 1; j++) {

				int rdval = rd.nextInt(len);

				while (rdval == maxIndex)
					rdval = rd.nextInt(len);

				sb.append(samplesStr[rdval]);
				sb.append("\\n");
			}
			sb.append(samplesStr[maxIndex]);
			newInst.setValue(1, sb.toString());
			newdata.add(newInst);
		}

		return newdata;
	}

	private Instance negativeBagSampling(Instance inst) {
		Instance bag = inst;

		String attrStr = bag.stringValue(1);
		String[] samplesStr = attrStr.split("\\n");
		int len = samplesStr.length;
		Random rd = new Random();

		Instance newInst = new Instance(inst.numAttributes());
		newInst.setDataset(inst.dataset());
		rd.setSeed(System.currentTimeMillis());

		Instances aftersampling = inst.attribute(1).relation()
				.stringFreeStructure();

		for (int j = 0; j < len; j++) {
			aftersampling
					.add(inst.relationalValue(1).instance(rd.nextInt(len)));
		}
		int relationValue;
		relationValue = newInst.attribute(1).addRelation(aftersampling);
		newInst.setValue(0, inst.value(0));
		newInst.setValue(1, relationValue);
		newInst.setValue(2, inst.classValue());

		return newInst;
	}

	private Instances negativeBagSampling(Instance inst, int times) {
		Instance bag = inst;
		Instances newdata = new Instances(inst.dataset(), 0);

		Attribute attr = bag.attribute(1);
		String attrStr = attr.toString();
		String[] samplesStr = attrStr.split("\\n");
		int len = samplesStr.length;
		Random rd = new Random();
		for (int i = 0; i < times; i++) {
			Instance newInst = new Instance(inst);
			rd.setSeed(System.currentTimeMillis());
			StringBuffer sb = new StringBuffer();
			int j = 0;
			for (; j < len - 1; j++) {
				sb.append(samplesStr[rd.nextInt(len)]);
				sb.append("\\n");
			}
			sb.append(samplesStr[rd.nextInt(len)]);
			newInst.setValue(1, sb.toString());
			newdata.add(newInst);
		}

		return newdata;
	}

	private double getPositiveProb(double[] sample) {
		double prob = 1.0;
		int numNegativeInsts = negativeBags.numInstances();
		for (int i = 0; i < numNegativeInsts; i++) {
			Instance current = negativeBags.instance(i);
			String attrStr = current.stringValue(1);
			String[] samplesStr = attrStr.split("\\n");
			int nlen = samplesStr.length;
			double dismax = 0.0;
			for (int k = 0; k < nlen; k++) {
				String nsampleStr = samplesStr[k];
				String[] valsStr = nsampleStr.split(",");
				double distance = 0.0;
				for (int q = 0; q < valsStr.length; q++) {
					double val = Double.parseDouble(valsStr[q]);
					distance += (sample[q] - val) * (sample[q] - val);
				}

				double temp = Math.exp((-1) * distance / (miu * miu));
				if (temp > dismax)
					dismax = temp;
			}

			prob = prob * (1 - dismax);
		}

		return prob;
	}

}
