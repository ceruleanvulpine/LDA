import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.EigenDecomposition;
import org.ejml.ops.CommonOps;


public class LdaData {
	
	DenseMatrix64F sqrt; //a square root of the covariance matrix, 
	//acquired thru eigendecomposition and with the eigenvalues square-rooted
	DenseMatrix64F means; //the means of each feature of each cluster
	int dim; //the number of features
	int clusters; //the number of clusters
	Random r; //to generate gaussian data
	double[] probabilities; //how likely is each cluster? 
	
	public LdaData(DenseMatrix64F cov, DenseMatrix64F mu, double[] probs){
		
		//cov = shared covariance matrix
		//means = means of each feature in each cluster
		//probs = probability of each cluster
		
		r = new Random();
		
		means = mu;
		probabilities = probs;
	
		//clusters = number of separate clusters
		clusters = probs.length;
		
		//dim = number of dimensions
		dim = cov.numCols;
		
		//make a copy of the covariance matrix to decompose, as decomposition changes it
		DenseMatrix64F covcopy = new DenseMatrix64F(cov.numCols, cov.numCols);
		CommonOps.insert(cov, covcopy, 0, 0);
		
		//decompose covcopy so you can access eigenvectors/values
		EigenDecomposition<DenseMatrix64F> e = DecompositionFactory.eig(cov.numCols, true, true);
		e.decompose(covcopy);
		
		//Q = eigenvectors, lambda = square roots of eigenvalues along the diagonal
		DenseMatrix64F Q = new DenseMatrix64F(cov.numRows,cov.numCols);
		DenseMatrix64F lambda = new DenseMatrix64F(cov.numRows,cov.numCols);
		
		for (int i = 0;i<cov.numRows;i++){
			lambda.set(i,i,Math.pow(e.getEigenvalue(i).real,.5));
			CommonOps.insert(e.getEigenVector(i), Q, 0, i);
		}
		
		//sqrt = q*lambda*q transpose
		DenseMatrix64F qtrans = new DenseMatrix64F(cov.numRows,cov.numRows);
		CommonOps.transpose(Q,qtrans);		
		DenseMatrix64F presqrt = new DenseMatrix64F(cov.numRows,cov.numRows);	
		CommonOps.mult(Q, lambda, presqrt);
		DenseMatrix64F squareroot = new DenseMatrix64F(cov.numRows,cov.numRows);
		CommonOps.mult(presqrt, qtrans, squareroot);
		sqrt = squareroot;
		
	}
	
	public DenseMatrix64F[] genData(int size){
		
		//generate (size) data points, assigned to clusters 
		//in proportion to "probs" 
		//with each cluster following the shared covariance matrix
		//return array of matrices where each matrix is a cluster
		DenseMatrix64F[] rtn = new DenseMatrix64F[clusters];
		
		//betterprobs is thresholds rather than fractional probs
		double[] betterprobs = new double[clusters];
		
		double runningsum = probabilities[0];
		
		for (int i=1;i<clusters;i++){
			betterprobs[i] = runningsum;
			runningsum = runningsum + probabilities[i];
		}
		
		//figure out how many data points each cluster will have
		int[] proportions = new int[clusters];
		
		for (int i=0;i<size;i++){
			int c=0;
			double rand = r.nextDouble();
			for (int j=0;j<clusters;j++){	
				if (rand<betterprobs[j]){break;}
				c=j;
			}
			proportions[c] = proportions[c] + 1;
		}
		
		
		//generate each cluster
		for(int i=0;i<clusters;i++){
			DenseMatrix64F cluster1 = new DenseMatrix64F(proportions[i],dim);
			
			//fill the matrix with generically gaussian data
			for (int a=0;a<proportions[i];a++){
				for (int b=0;b<dim;b++){
					cluster1.set(a,b,r.nextGaussian());
				}
			}
			
			//multiply by sqrt to transform to the data with the right covariance matrix
			DenseMatrix64F cluster2 = new DenseMatrix64F(proportions[i],dim);
			CommonOps.mult(cluster1,sqrt,cluster2);
	
			//add appropriate means
			for (int a=0;a<proportions[i];a++){
				for (int b=0;b<dim;b++){
					cluster2.add(a, b, means.get(i,b));
				}
			}
			
			rtn[i] = cluster2;
		}
		
		return rtn;

	}

}
