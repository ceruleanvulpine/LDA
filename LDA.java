import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;


public class LDA {

	double[] probhat; //estimated probability of each cluster
	DenseMatrix64F muhat; //estimated means for each feature and cluster
	DenseMatrix64F covhat; //estimated covariance matrix
	
	public LDA(DenseMatrix64F[] data){
		//given data, the LDA constructor
		//stores estimated probabilities, means and covariance 
		//as fields that can be used to generate linear discriminant values
		
		//estimate clusters' probabilities by fraction of data
		probhat = new double[data.length];
		double total = 0;
	
		for (int i=0;i<data.length;i++){
			total = total + data[i].numRows;
			probhat[i] = data[i].numRows;
		}
		
		for (int i=0;i<probhat.length;i++){
			probhat[i] = probhat[i]/total;
		}
		
		/*
		
		means:
		
		[-dimension (j)- ]
		|        ^       |
		| #clusters (i)  |
		|        |       |
		[        \/      ]
		
		each cluster:

		[-dimension (j)- ]
		|        ^       |
		|cluster size (k)|
		|        |       |
		[        \/      ]
		
		*/
		
		muhat = new DenseMatrix64F(data.length,data[0].numCols);
		double cursum = 0;
		//estimate means by average
		for (int i=0;i<data.length;i++){
			for (int j=0;j<data[i].numCols;j++){
				cursum = 0;
				for (int k=0;k<data[i].numRows;k++){
					cursum = cursum + data[i].get(k,j);
				}
				muhat.set(i,j,cursum/(double)data[i].numRows);
			}
		}
		
		//estimate covariance 
		
		covhat = new DenseMatrix64F(data[0].numCols,data[0].numCols);
		cursum = 0;
		int totalsize = 0;
		for (int i=0;i<data.length;i++){
			totalsize = totalsize+data[i].numRows;
		}
		
		for (int i=0;i<data[0].numCols;i++){
			for (int j=0;j<data[0].numCols;j++){
				
				//for each i,j in the matrix
				//add every data point in every cluster's
				//centralized ith feature * centralized jth feature
				//then divide by n-1
				
				cursum = 0;
				
				for (int k=0;k<data.length;k++){
					for (int l=0;l<data[k].numRows;l++){
						
						cursum = cursum + (data[k].get(l,i)-muhat.get(k,i))*(data[k].get(l,j)-muhat.get(k,j));
						
					}
				}
				
				cursum = cursum/(double)(totalsize-1);
				covhat.set(i,j,cursum);
			}
		}	
		
	}
	
	
	public double[] deltahat(double[] data){
		
		//given a new data point, this method returns all of its linear discriminant values
		//according to the equation given in the book.
		//the highest value corresponds to the cluster
		//this data point is predicted to belong to
		
		double[] rtn = new double[probhat.length];
		DenseMatrix64F data2 = new DenseMatrix64F(1,data.length);
		
		for (int k=0;k<data.length;k++){
			data2.set(0,k,data[k]);
		}
		
		
		DenseMatrix64F mu_k = new DenseMatrix64F(data.length,1);
		DenseMatrix64F mu_k_t = new DenseMatrix64F(1,data.length);
		DenseMatrix64F invcov = new DenseMatrix64F(covhat.numCols,covhat.numCols);
		CommonOps.invert(covhat,invcov);
		
		DenseMatrix64F step1 = new DenseMatrix64F(1,data.length);
		DenseMatrix64F step2 = new DenseMatrix64F(1,1);
		
		double intersum;
		for (int k=0;k<rtn.length;k++){
			
			intersum=0;
			
			mu_k_t = CommonOps.extractRow(muhat,k,mu_k_t);
			CommonOps.transpose(mu_k_t,mu_k);

			CommonOps.mult(data2, invcov, step1);
			CommonOps.mult(step1,mu_k,step2);
			
			intersum = intersum + step2.get(0,0);
			
			CommonOps.mult(mu_k_t, invcov, step1);
			CommonOps.mult(step1, mu_k, step2);
			
			intersum = intersum - 0.5*step2.get(0,0);
			
			intersum = intersum + Math.log(probhat[k]);
			
			rtn[k] = intersum;
		}
		
		return rtn;
		
	}
	
	public double[] hypcoeff(int k, int j){
		
		//this method returns the coefficients of 
		//the hyperplane dividing the k-region from the j-region
		// ax1 + bx2 + cx3 .. = 0
		
		double[] rtn = new double[covhat.numCols+1];

		DenseMatrix64F invcov = new DenseMatrix64F(covhat.numCols,covhat.numCols);
		CommonOps.invert(covhat,invcov);
		
		/*
		
		means:
		
		[-dimension (j)- ]
		|        ^       |
		| #clusters (i)  |
		|        |       |
		[        \/      ]
		
		mu_k is a column vector with dimension rows
		mu_k_t is a row vector with dimension columns
		
		*/
		
		DenseMatrix64F mu_k = new DenseMatrix64F(covhat.numCols,1);
		DenseMatrix64F mu_j = new DenseMatrix64F(covhat.numCols,1);
		
		DenseMatrix64F mu_k_t = new DenseMatrix64F(1,covhat.numCols);
		DenseMatrix64F mu_j_t = new DenseMatrix64F(1,covhat.numCols);
		
		mu_k_t = CommonOps.extractRow(muhat,k,mu_k_t);
		mu_j_t = CommonOps.extractRow(muhat,j,mu_j_t);
		
		CommonOps.transpose(mu_k_t,mu_k);
		CommonOps.transpose(mu_j_t,mu_j);
		
		DenseMatrix64F mu_k_minus_j = new DenseMatrix64F(covhat.numCols,1);
		CommonOps.subtract(mu_k, mu_j, mu_k_minus_j);
		
		DenseMatrix64F currow = new DenseMatrix64F(1,covhat.numCols);
		DenseMatrix64F product = new DenseMatrix64F(1,1);
		
		for (int i=0;i<covhat.numCols;i++){
			currow = CommonOps.extractRow(invcov,i,currow);
			CommonOps.mult(currow, mu_k_minus_j, product);
			rtn[i] = product.get(0,0);
		}
		
		DenseMatrix64F step1 = new DenseMatrix64F(1,covhat.numCols);
		DenseMatrix64F step2 = new DenseMatrix64F(1,1);
		
		CommonOps.mult(mu_k_t,invcov,step1);
		CommonOps.mult(step1,mu_k,step2);
		
		rtn[rtn.length-1] = step2.get(0,0)*-.5;
		
		CommonOps.mult(mu_j_t,invcov,step1);
		CommonOps.mult(step1,mu_j,step2);
		
		rtn[rtn.length-1] = rtn[rtn.length-1] + step2.get(0,0)*.5;
		
		rtn[rtn.length-1] = rtn[rtn.length-1] + Math.log(probhat[k]/probhat[j]);
		
		return rtn;
		
	}
	
}
