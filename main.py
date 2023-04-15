
#Author - R <> 

import sys
import os 
import argparse
from pathlib import Path 

import pandas as pd  
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_val_score, KFold

results_path = Path('results/')
data_path    = Path('data/')

assert all(list(map(lambda x: os.path.exists(x),(results_path, data_path)))),"Check that results/ and data/ path exist"

# lambda helper 
mse = lambda x: (x**2).mean() # where x must be the vector of residuals
z_score = lambda x: (x - x.mean())/x.std()

## functions
def exists(x):
    return x is not None 

def default(val,d):
    return val if exists(val) else d 

def verbose_read(path):
    """
    Read in file for the question, if error then raise. 

    Args:
        path (_type_): str path to file

    Returns:
        _type_: dataframe for QX 
    """
    try:
        df = pd.read_csv(path)
        return df 
    except Exception as e:
        print(f"Error: {e}")

def relative_pe_by_sector(df):
    """Get industry sector median for each company

    Args:
        df (DataFram):  of companies, group, and pe ratio

    Returns:
        DataFrame: With added PE_RATIO_REL column
    """

    # dont edit original
    df = df.copy()

    # Calculate median P/E ratio for each sector
    median_pe = df.groupby('INDUSTRY_SECTOR')['PE_RATIO'].median()
    
    # lookup industry sector median that corresponds to each row 
    median_pe_values = median_pe[df['INDUSTRY_SECTOR']].values 
    
    # Calculate each company's relative % P/E premium/discount over/below sector median
    df['PE_RATIO_REL'] = np.divide(np.subtract(df['PE_RATIO'], median_pe_values),median_pe_values)*100
    
    return df
 
def get_model_scores(X,y,kf,models, names = ["OLS","Random Forest"]):
    """lambda map  models (LR and RF) to the dataset using kfold cross validation

    Args:
        models (list): list of models to use
        names (list, optional): allows referencing model name when looking at test mse of kfold cross validation. Defaults to ["Random Forest","OLS"].

    Returns:
        DataFrame: Fold averaged test MSE for each model
    """

    scores = pd.DataFrame(map(lambda model: cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf),models)).rename(index={0: names[0], 1: names[1]})
    return  -scores.mean(1)

## classes
class files:
        
        """File class to hold str of file names for each question"""
        
        data_path = default(data_path, Path(''))
        FILE_Q1 = data_path/"python munging.csv"
        FILE_Q2 = data_path/"PE_ratio.csv"
        FILE_Q3 = data_path/"regression.csv"
        FILE_Q4 = data_path/"boston.csv"

class plotter:

    """class for organizing plot functions"""
    
    def pretty_correlation_plot(data,title = ''):
        """Given a DataFram of variables plot the correlation matrix among all variables. Use a mask to visualize only the lower triangular entries. 

        Args:
            data (DataFram): Varialbes to run correlation analysis on 
        """
        # Generate and visualize the correlation matrix
        corr = data.corr().round(2)

        # Mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set figure size
        f, ax = plt.subplots(figsize=(20, 20))

        # Define custom colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap
        sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        plt.title(title)

        plt.tight_layout()
        plt.show()
    
    def cooks_plot(cooks_distance):
                # Plot Cook's distance values
                fig, ax = plt.subplots(figsize=(12,6))
                ax.stem(cooks_distance, markerfmt=",", use_line_collection=True)
                ax.set_xlabel("Observation number")
                ax.set_ylabel("Cook's distance")
                ax.set_title("Cook's distance plot")
                plt.show()

    def target_hist(y):
            """_summary_

            Args:
                y (_type_): _description_
            """
            # a lot of 50 outliers 
            sns.set_style("whitegrid")
            sns.distplot(y, kde=False, color='b', hist_kws={'alpha': 0.8})
            plt.xlabel('Price ($1000s)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Boston Housing Prices')
            plt.show()
       
    def pe_by_industry_hist(df):
        """_summary_
        """
        
        # Create facet grid plot
        g = sns.FacetGrid(df,  col='INDUSTRY_SECTOR',col_wrap=3, sharex=False, sharey=True, height=4)
        g.map(sns.histplot, 'PE_RATIO')
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("P/E Ratio", "Count")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('Distribution of P/E Ratios by Industry')
        plt.show()

    def graph(formula, x_range, label=None):
        """
        Helper function for plotting cook's distance lines
        """
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')

    def diagnostic_plots(X, y, model_fit=None):
        """
        Function to reproduce the 4 base plots of an OLS model in R.
        ---
        Inputs:

        X: A numpy array or pandas dataframe of the features to use in building the linear regression model

        y: A numpy array or pandas series/dataframe of the target variable of the linear regression model

        model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be
                                generated from X, y
        
        taken from https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/
        """
        from statsmodels.graphics.gofplots import ProbPlot

        if not model_fit:
            model_fit = sm.OLS(y, sm.add_constant(X)).fit()

        # create dataframe from X, y for easier plot handling
        dataframe = pd.concat([X, y], axis=1)

        # model values
        model_fitted_y = model_fit.fittedvalues
        # model residuals
        model_residuals = model_fit.resid
        # normalized residuals
        model_norm_residuals = model_fit.get_influence().resid_studentized_internal
        # absolute squared normalized residuals
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
        # absolute residuals
        model_abs_resid = np.abs(model_residuals)
        # leverage, from statsmodels internals
        model_leverage = model_fit.get_influence().hat_matrix_diag
        # cook's distance, from statsmodels internals
        model_cooks = model_fit.get_influence().cooks_distance[0]

        plot_lm_1 = plt.figure()
        plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe,
                                    lowess=True,
                                    scatter_kws={'alpha': 0.5},
                                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

        plot_lm_1.axes[0].set_title('Residuals vs Fitted')
        plot_lm_1.axes[0].set_xlabel('Fitted values')
        plot_lm_1.axes[0].set_ylabel('Residuals');

        # annotations
        abs_resid = model_abs_resid.sort_values(ascending=False)
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3.index:
            plot_lm_1.axes[0].annotate(i,
                                        xy=(model_fitted_y[i],
                                            model_residuals[i]));

        QQ = ProbPlot(model_norm_residuals)
        plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
        plot_lm_2.axes[0].set_title('Normal Q-Q')
        plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
        plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            plot_lm_2.axes[0].annotate(i,
                                        xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                            model_norm_residuals[i]));

        plot_lm_3 = plt.figure()
        plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
        sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
        plot_lm_3.axes[0].set_title('Scale-Location')
        plot_lm_3.axes[0].set_xlabel('Fitted values')
        plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_norm_resid_top_3:
            plot_lm_3.axes[0].annotate(i,
                                        xy=(model_fitted_y[i],
                                            model_norm_residuals_abs_sqrt[i]));


        plot_lm_4 = plt.figure();
        plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
        sns.regplot(model_leverage, model_norm_residuals,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
        plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
        plot_lm_4.axes[0].set_ylim(-3, 5)
        plot_lm_4.axes[0].set_title('Residuals vs Leverage')
        plot_lm_4.axes[0].set_xlabel('Leverage')
        plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

        # annotations
        leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
        for i in leverage_top_3:
            plot_lm_4.axes[0].annotate(i,
                                        xy=(model_leverage[i],
                                            model_norm_residuals[i]));

        p = len(model_fit.params) # number of model parameters
        plotter.graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
                np.linspace(0.001, max(model_leverage), 50),
                'Cook\'s distance') # 0.5 line
        plotter.graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
                np.linspace(0.001, max(model_leverage), 50)) # 1 line
        plot_lm_4.legend(loc='upper right');

## main
def main(args):

    assert exists(args.question), "Question param must be defined"
    assert args.question in [1,2,3,4,5], "Value of q is not in scope of Q1-5, should be one of [1,2,3,4,5]"

    if args.question == 1:
        # Q1
        df                        = verbose_read(files.FILE_Q1)
        includeCouponList         = [100,500]
        InstruementTypeToOverride = args.InsOverrideList
        logic = lambda x: x.apply(lambda x: f"{str(x['CMATicker'])}-{str(int(x['Tenor']))}" if x['InstrumentType']  in InstruementTypeToOverride else x['ClientEntityId'], axis=1)

        # assign new primary id 
        out = df.assign(PRIMARY_ID = lambda x: logic(x)).loc[lambda x: x.Coupon.isin(includeCouponList) ].reset_index(drop = True)

        print(out.describe())
        print(out.head())

        assert ~out.PRIMARY_ID.isna().any(), "Some IDs are NaN"

        if args.save:
            out.to_excel(results_path/"Q1Results.xlsx",index = False)
    
    if args.question == 2:
        # Q2
        df = verbose_read(files.FILE_Q2) 
        
        print(df.describe())
        print(df.head())
        
        # make sure PE ratio is in a useable format, if not also take a look at values for non float pe ratios
        if ~isinstance(df.PE_RATIO,float): 
            temp = df.copy()
            df['PE_RATIO'] = pd.to_numeric(df['PE_RATIO'], errors='coerce') 
            nan_indices = df[df['PE_RATIO'].isna()].index
            print(temp.iloc[nan_indices])
     
        # A - For each sector, show descriptive statistics of P/E ratios
        print(df.groupby(['INDUSTRY_SECTOR'])['PE_RATIO'].agg({'mean','median','std','min','max',('spread', lambda x: x.max() - x.min()) }))# .describe()

        # B - Plot the distribution of P/E ratios for each sector in one chart
        plotter.pe_by_industry_hist(df)

        # C - What is each company’s relative % P/E premium/discount over/below sector median? Add that to dataframe.
        df_enhanced = relative_pe_by_sector(df)

        if args.save:
            df_enhanced.to_excel(results_path/"Q2Results.xlsx",index = False)

    if args.question == 3:
        # Q3
        data     = pd.read_csv(files.FILE_Q3)
        dataLen  = len(data)
        splitInt = 75 # First X rows for training, rest for testing 
        varSet   = default(args.varSet,['0', '2', '4', '5', '6', '9'])

        plotter.pretty_correlation_plot(data)

        if "Tr" in args.steps: # Tr-aining
            # Define the independent and dependent variable
            X = data.get(varSet).head(splitInt)
            y = data.get(['y']).head(splitInt)

            # Add a constant to the independent variables
            X = sm.add_constant(X)

            # Fit the linear model 
            model = sm.OLS(y, X).fit()

            # Calculate Cook's distance for each observation
            influence = model.get_influence()
            cooks_distance = influence.cooks_distance[0]
            plotter.cooks_plot(cooks_distance)

            # Print the summary of the model
            print(model.summary())

            # view stats
            resid_df = y.join(pd.Series(model.fittedvalues,name='y_hat'))
            resid_df.plot()
            train_mse = mse(np.subtract(y.values.squeeze(), model.fittedvalues.values))
            
            plotter.diagnostic_plots(X, y, model_fit=default(model,None))

        if "Te" in args.steps: # Te-sting
            # testing
            Xtest = data.get(varSet).tail(dataLen - splitInt)
            ytest = data.get(['y']).tail(dataLen - splitInt)

            yhat_test = model.predict(sm.add_constant(Xtest))
            resid_df_test = ytest.join(pd.Series(yhat_test,name='y_hat_test')).assign(resid = lambda x: x.y - x.y_hat_test)
            # resid_df_test.plot()

            # Residual plot evenly distributed 
            plt.scatter(resid_df_test.y_hat_test, resid_df_test.resid)

            test_mse = mse(np.subtract(ytest.values.squeeze(), yhat_test.values))
            
            test_data_full = data.get(varSet).merge(resid_df_test, how = 'inner', left_index=True, right_index = True, validate = '1:1')
            
            # Check endogenity, residual correlation with predictors 
            plotter.pretty_correlation_plot(test_data_full,title = "Exogeneity Test: resid correlation to predictors")

            print(f"train mse of {train_mse:.3f} vs test mse of {test_mse:.3f}")
    
    if args.question == 4:
        # Q4 
        df = verbose_read(files.FILE_Q4)
        
        if args.remove_outliers:
            df = df.loc[lambda x: (x.medv < 50)] # .assign(medv_z = lambda x: z_score(x.medv)).loc[lambda x: abs(x.medv_z) < 3]
        
        # explore data 
        print(df.isna().sum())
        print(df.describe())
        print(df.head())

        plotter.pretty_correlation_plot(df)

        # Separate the target variable and the input features
        y = df["medv"]
        X = df.drop("medv", axis=1)

        plotter.target_hist(y)

        # Define the models
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        lr = LR()

        # Define the cross-validation object
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # if modelslist not provided, use linear reg, and random forest 
        scores = get_model_scores(X = X,y = y,kf = kf,models = default(args.modelList,(lr,rf)))

        print(scores)
        if args.save:
            scores.to_excel(results_path/"Q3Results.xlsx",index = False)
  
    if args.question == 5:
        # The date format used in the cds_prices table is an integer representing the number of days since January 1, 1970 (also known as Unix time). 
        query1 = """SELECT *
                FROM cds_prices
                WHERE Date = (SELECT MAX(Date) FROM cds_prices);
                """

        # Tenor_prices column in the cds_prices table is defined as decimal(4,2), thus it can store a maximum of 4 digits, with 2 digits after the decimal point.
        # Not clear if we need to filter out Nulls, assume we dont to align with directive of prompt
        query2 = """SELECT cp.Date, cm.CMATicker, cp.ParSpreadMid
                FROM cds_prices cp
                JOIN cds_mapping cm ON cp.secID = cm.secID
                WHERE cm.CMATicker = 'RUSSIA'
                AND cm.Tenor_mapping = 5.00
                AND cp.Date >= 20140101;"""
        
        print(query1, query2)

sys.argv = [''] # uncomment if not running from cli
parser = argparse.ArgumentParser()

parser.add_argument("-q", "--question", type=int, default=5,help="question parameter (default: 1)")
parser.add_argument("-s", "--steps", type=str, default="TrTe",help="steps parameter for Q3 (default: 'TeTr')")
parser.add_argument("-I", "--InsOverrideList", type=list, default=['Index','Tranche'],help="Instrument overrid list for Q1 (default: 'Index','Tranche')")
parser.add_argument("-m", "--modelList", type=tuple, default=None,help="Tuple of models to evaluate in Q4")
parser.add_argument("-v", "--varSet", type=list, default=None,help="List of variables to use in Q3")
parser.add_argument("-sa", "--save", type=bool, default=False,help="Bool if save output (if applicable) of Qs to results location")
parser.add_argument("-r", "--remove_outliers", type=bool, default=False,help="Bool if to remove outliers in Q4 for Boston housing dataset. ")

args = parser.parse_args()


if __name__=='__main__':
    main(args) 


import torch
from einops import rearrange

input_dim = 4
weights = torch.randn(input_dim // 2)
x = torch.randn(2, input_dim)


# reshape weights to have a leading singleton dimension
weights_reshaped = rearrange(weights, 'd -> 1 d')

# broadcast weights_reshaped along the first dimension to match x
weights_broadcasted = weights_reshaped.expand(x.shape[0], -1)

# multiply x and weights_broadcasted
freqs = x * weights_broadcasted * 2 * math.pi

print(freqs.shape) # output: torch.Size([2, 2])
