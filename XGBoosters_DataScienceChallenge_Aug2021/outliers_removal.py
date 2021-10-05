
class Out_Remove:
    def remove_outliers(x):

        ## storing names of columns in Column_Names
        Column_Names=x.columns

        ## importing matplotlib and seaborn for visualisation of outliers
        import matplotlib.pyplot as plt
        import seaborn as sns   

        ## iterating through Column_Names using try and except for distinguishing between numerical and categorical columns
        for j in Column_Names:
            try:
                print('Before Removing Outliers')

                ##visualisation of outliers
                a = sns.boxplot(data=x,x=x[j])
                plt.tight_layout() 
                plt.show() 

                xy=x[j]    
                mydata=pd.DataFrame()

                updated=[]
                Q1,Q3=np.percentile(xy,[25,75])
                IQR=Q3-Q1
                minimum=Q1-1.5*IQR
                maximum=Q3+1.5*IQR

                ## using the maximum and minimum values obtained from quartiles and inter-quartile range
                ## any outliers greater than maximum are updated to be equal to maximum
                ## any outliers lesser than minimum are updated to be equal to minimum
                ## here, no outliers have been removed to prevent loss of data

                for i in xy:
                    if(i>maximum):
                        i=maximum
                        updated.append(i)
                    elif(i<minimum):
                        i=minimum
                        updated.append(i)
                    else:
                        updated.append(i)

                x[j]=updated
                print('After Removing Outliers')

                ## visualising after removing outliers
                b= sns.boxplot(data=x,x=x[j])
                plt.tight_layout() 
                plt.show()

            except:
                continue

        return x
