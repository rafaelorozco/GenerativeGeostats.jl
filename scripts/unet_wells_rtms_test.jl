#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=20G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=15G srun --pty julia 

using DrWatson
@quickactivate "GenerativeGeostats.jl"
import Pkg; Pkg.instantiate()

#using Pkg;Pkg.add(PackageSpec(url="https://github.com/mloubout/UNet.jl", rev="remove-bad-layes")) #SUPER SMOOTH prior
#using Pkg;Pkg.add(PackageSpec(url="https://github.com/mloubout/UNet.jl", rev="no_tanh"))
using Flux, UNet, Zygote
using PyPlot
using JLD2, BSON
using Statistics, Random, LinearAlgebra
using ImageQualityIndexes

function get_batch_well(X; width_well = 3, n_wells = 1,min_dist = 20 )
	Y = zeros(Float32,size(X))
	nx = size(X)[1]
	ny = size(X)[2]
	ny_half = Int(round(div(ny,2)))
	for b in 1:size(X)[end]
		set_possible = collect(1:(nx-width_well+1))
		wells = []
		n_wells_i = n_wells
		#Randomly place less wells
		# if rand() > 0.5
    	# 	n_wells_i = Int(round(div(n_wells,2)))
	  	# end
		for i in 1:n_wells_i
			sel_rand = rand(set_possible)
			append!(wells,sel_rand:(sel_rand+width_well-1))
			deleteat!(set_possible, findall(x->x in (sel_rand-min_dist):(sel_rand+min_dist+width_well-1),set_possible))
			#end_well = rand(ny-ny_half:ny) # random depth
			end_well = ny
			Y[sel_rand:(sel_rand+width_well-1),1:end_well,:,b] .= X[sel_rand:(sel_rand+width_well-1),1:end_well,:,b]
		end
	end
	Y
end

# Plotting configs
save_net_path = "/slimdata/rafaeldata/savednets_krig_hint2"
plot_path = "/slimdata/rafaeldata/plots/unsuper_krig/unet-wells-rtm"

# Training hyperparameters 
device = gpu

# Get data 
data_path = "/slimdata/rafaeldata/fwiuq_eod/rtms_normalized.jld2"
m_train = JLD2.jldopen(data_path, "r")["m_train"];
rtm_train = JLD2.jldopen(data_path, "r")["rtm_train"];

Xs = m_train ./ 5f0

# number of training examples 
i_train = 200

Random.seed!(122);
n_wells = 2
width_well = 3 #in pixels
noise_wells = 0.0 #additive noise to wells from ground truth velocity model
min_dist = 100
Xs_wells = get_batch_well(Xs + noise_wells.*randn(size(Xs)); width_well = width_well, n_wells=n_wells, min_dist = min_dist );

inds_train = range(1, 1000; step=div(1000, i_train))
X_train = Xs_wells[:,:,:,inds_train]
Y_train = rtm_train[:,:,:,inds_train]
Xs      = Xs[:,:,:,inds_train]


#Load in pretrained unet
net_train =  "batch_size=8_clipnorm_val=3.0_e=100_lr=0.0005_min_dist=100_n_train=192_n_x_tr=400_nx=512_testmode_a=true_unet_lev=4.bson"
unet_cpu = BSON.load(joinpath(save_net_path,net_train))["unet_cpu"]
unet = unet_cpu |> device

testmode_a = BSON.load(joinpath(save_net_path,net_train))["testmode_a"]
testmode_a && testmode!(unet, true);
inds = [10,50]
num_cols = 5
plots_len = length(inds)
for ind in inds
	conds = Y_train[:, :, :, ind:ind];
	X_gt  = Xs[:,:,:,ind:ind] |> cpu
	Y     = X_train[:,:,:,ind:ind] |> cpu

	X_post = unet(conds|> device) |> cpu
	x_hat = X_post[:,:,1,1]	

	error_mean = abs.(x_hat-X_gt[:,:,1,1])
	ssim_i = round(assess_ssim(x_hat, X_gt[:,:,1,1]),digits=2)
	rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)

	y_plot = conds[:,:,1,1]' |> cpu
	a = quantile(abs.(vec(y_plot)), 98/100)

	vmax_v = maximum(X_gt); vmin_v = minimum(X_gt)

	fig = figure(figsize=(20, 5)); 
	subplot(plots_len, num_cols,1); imshow(y_plot, vmin=-a,vmax=a,interpolation="none", cmap="gray")
	axis("off"); title("rtm");#colorbar(fraction=0.046, pad=0.04);

	subplot(plots_len, num_cols,2); imshow(Y[:,:,1,1]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="terrain")
	axis("off"); title(L"$\mathbf{y} = \mathbf{F}(\mathbf{x^{*}})$ observation"); #colorbar(fraction=0.046, pad=0.04);

	subplot(plots_len, num_cols,3); imshow(X_post[:,:,1,1]',vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="terrain")
	axis("off");  #colorbar(fraction=0.046, pad=0.04);
	title("Posterior samp G(z;rtm)")

	subplot(plots_len, num_cols,4); imshow(X_gt[:,:,1,1]',  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="terrain")
	axis("off"); title(L"Reference $\mathbf{x^{*}}$") ;# colorbar(fraction=0.046, pad=0.04)

	subplot(plots_len, num_cols,5); imshow(error_mean' , vmin=0,vmax=nothing, interpolation="none", cmap="magma")
	axis("off");title("RMSE "*string(rmse_i)) ; #cb = colorbar(fraction=0.046, pad=0.04)

	tight_layout()
	fig_name = @strdict  ind
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)
end
	
		