using DataFrames
using CSV

savepath = "./data/sushi3-2016/train.csv"

# Load datasets
df_idata = CSV.read("./data/sushi3-2016/sushi3.idata", header=false, delim='\t')
df_udata = CSV.read("./data/sushi3-2016/sushi3.udata", header=false, delim='\t')
df_score = CSV.read("./data/sushi3-2016/sushi3b.5000.10.score", header=false, delim=' ')

df_train = DataFrame(user_id = df_udata[1])
for j in 1:length(df_udata[1])
    df = copy(df_idata)
    df[:user_id] = df_udata[1][j]
    df = hcat(df,convert(DataFrame,convert(Array,df_score[j, :])'))

    global df_train
    df = df[df[:x1].>-1,:]
    if j == 1
        df_train = join(df_train,df,on=:user_id)
    else
        df_train = vcat(df_train,df)
    end
end
delete!(df_train,:Column2)
rename!(df_train,:x1 => :score)

rename!(df_udata,:Column1 => :user_id)
df_train = join(df_udata,df_train,on=:user_id)

df_train |> CSV.write(savepath,delim=' ',writeheader=false)
