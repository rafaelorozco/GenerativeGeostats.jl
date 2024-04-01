#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=20G srun --pty julia 

using DrWatson
@quickactivate "GenerativeGeostats.jl"
import Pkg; Pkg.instantiate()

using Flux, UNet, Zygote
using PyPlot
using JLD2
using Statistics, Random, LinearAlgebra
using ImageQualityIndexes

function get_batch_well(X; width_well = 3, n_wells = 1, min_dist = 20 )
	Y = zeros(Float32,size(X))
	Y_mask = zeros(Float32,size(X))
	Y_cond = zeros(Float32,size(X))
	nx = size(X)[1]
	ny = size(X)[2]
	for b in 1:size(X)[end]
		set_possible = collect(1:(nx-width_well+1))
		wells = []

		for i in 1:n_wells
			sel_rand = rand(set_possible)
			append!(wells,sel_rand:(sel_rand+width_well-1))
			deleteat!(set_possible, findall(x->x in (sel_rand-min_dist):(sel_rand+min_dist+width_well-1),set_possible))
			end_well = ny
			Y[sel_rand:(sel_rand+width_well-1),1:end_well,:,b] .= X[sel_rand:(sel_rand+width_well-1),1:end_well,:,b]
		end
	end
	Y
end

function get_cm_l2_ssim(G, Xs_batch, Y_batch; device=gpu, num_samples=1)
	    num_test = size(Y_batch)[end]
	    l2_total = 0 
	    ssim_total = 0 
	    #get cm for each element in batch
	    for i in 1:num_test
			conds = Y_batch[:, :, :, i:i];
			x_gt  = Xs_batch[:,:,1,i] |> cpu
			x_hat = G(conds|> device )[:,:,1,1] |>cpu
	    	ssim_total += assess_ssim(x_hat, x_gt)
			l2_total   += sqrt(mean((x_hat - x_gt).^2))
		end
	return l2_total / num_test, ssim_total / num_test
end

# Training hyperparameters 
device = gpu

lr      = 1f-4
noise_lev_x = 0.0f0

batch_size   = 8#
n_epochs     = 800

save_every   = 100
plot_every   = 5

# Define data directory CHANGE TO YOUR PATH
save_net_path = "/slimdata/rafaeldata/savednets_krig_hint2"
plot_path = "/slimdata/rafaeldata/plots/unsuper_krig/unet-wells"
data_path = "/slimdata/rafaeldata/fwiuq_eod/rtms_normalized.jld2"

if !isfile(data_path)
println("downloading")
   run(`wget "https://www.dropbox.com/scl/fi/htnw7io4xe6vdedtex23m/rtms_normalized.jld2?rlkey=dtb2hqoyt6vx5ypyfqutwcm6v&dl=0" -q -O $data_path`)
end

m_train = JLD2.jldopen(data_path, "r")["m_train"];
m_train = m_train ./ 5f0

#resize to jesus dataset just to confirm size works
using Images
Xs = zeros(Float32,600,200,1,size(m_train)[end])
for i in 1:size(m_train)[end]
	Xs[:,:,:,i] = imresize(m_train[:,:,1,i], (600, 200))
end

# training indexes 
i_train = 400

Random.seed!(122);
n_wells = 7
width_well = 3
noise_wells = 0.0
min_dist = 25
Xs_wells = get_batch_well(Xs + noise_wells.*randn(size(Xs)); width_well = width_well, n_wells=n_wells, min_dist = min_dist );

inds_train = range(1, 1000; step=div(1000, i_train)) #just to get more variability in datset. Can take away
X_train  = Xs_wells[:,:,:,inds_train]
Xs = Xs[:,:,:,inds_train]

n_c  = size(X_train)[end-1] 
nx, ny, n_in, n_train = size(X_train)
N = nx*ny

n_batches      = cld(n_train, batch_size)-1
n_train        = batch_size*n_batches

unet_lev = 5
unet = Unet(n_c,n_in,unet_lev)|> device;
trainmode!(unet, true) ;
ps = Flux.params(unet);

# Optimizer
opt = ADAM(lr)

# Training logs 
loss   = []; ssim   = []; l2_cm  = [];
loss_test   = []; ssim_test   = []; l2_cm_test  = [];

function loss_unet(Y, Y_curr;)
	mask = Y .> 0
	X_gen = unet(Y_curr |> device;)
	ΔX = (mask .* X_gen .- Y) 
	norm(ΔX)^2
end

chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]

#probability of adding extra corruption
corr_prob = 0.25
corr_prob_pixel = 0.25

#size of random data aug crops
n_x_tr = 256
n_y_tr = 192

for e=1:n_epochs # epoch loop
	idx_e = reshape(randperm(n_train), batch_size, n_batches) 
    for b = 1:n_batches # batch loop
    	@time begin

		Y_pre = X_train[:, :, :, idx_e[:,b]] 
		Y = zeros(Float32,n_x_tr,n_y_tr,1,batch_size);
		Y_corr = zeros(Float32,n_x_tr,n_y_tr,1,batch_size);

		for i in 1:batch_size
			ran_x = rand(collect(1:(nx-n_x_tr)))
			ran_y = rand(collect(1:(ny-n_y_tr)))
	    	Y[:,:,:,i:i] = Y_pre[ran_x:ran_x+n_x_tr-1,ran_y:ran_y+n_y_tr-1,:,i:i]
	    	Y_corr[:,:,:,i:i] = Y_pre[ran_x:ran_x+n_x_tr-1,ran_y:ran_y+n_y_tr-1,:,i:i]
			if rand() > 0.5
	    		Y[:,:,:,i:i] = Y[end:-1:1,:,:,i:i]
	    		Y_corr[:,:,:,i:i] = Y_corr[end:-1:1,:,:,i:i]
		  	end

		  	inds_wells = findall(x->x!=0, Y[:,1,1,i])
		  	inds_wells = chunk(inds_wells,width_well)
		  	for ind_set in inds_wells
		  		if rand() > corr_prob
		    		Y_corr[ind_set,:,:,i:i] .= 0
			  	end
			end

			inds_rand = findall(x->x!=0, Y[:,:,1,i])
			for ind_set in inds_rand
		  		if rand() > corr_prob_pixel
		    		Y_corr[ind_set,:,i:i] .= 0
			  	end
			end
	    end

	  	f, back = Zygote.pullback(() -> loss_unet(Y|> device, Y_corr|> device;), ps);
    
		gs = back(one(f))
		Flux.update!(opt, ps, gs)

	    append!(loss, f / (batch_size))
	    print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	        "; f l2 = ",  loss[end],"\n")
	    Base.flush(Base.stdout)
		end
    end

    if(mod(e,plot_every)==0)
    	# get conditional mean metrics over training batch  
	    @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(unet, Xs, X_train; device=device)
	    append!(ssim, cm_ssim_train)
	    append!(l2_cm, cm_l2_train)

		num_cols = 5
	    plots_len = 2

	    inds = [10,50]
	    for i in 1:plots_len
	    	ind = inds[i]

			X_gt  = Xs[:,:,:,ind:ind] |> cpu
			Y     = X_train[:,:,:,ind:ind] |> cpu

			X_post = unet(Y|> device) |> cpu
			x_hat = X_post[:,:,1,1]	

			error_mean = abs.(x_hat-X_gt[:,:,1,1])
			ssim_i = round(assess_ssim(x_hat, X_gt[:,:,1,1]),digits=2)
			rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)

			y_plot = Y[:,:,1,1]' |> cpu
			a = quantile(abs.(vec(y_plot)), 98/100)

			vmax_v = maximum(Xs); vmin_v = minimum(Xs)

			fig = figure(figsize=(20, 5)); 
			subplot(plots_len, num_cols,1); imshow(Y[:,:,1,1]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="terrain")
			axis("off"); title(L"$\mathbf{y} = \mathbf{F}(\mathbf{x^{*}})$ observation"); #colorbar(fraction=0.046, pad=0.04);

			subplot(plots_len, num_cols,2); imshow(X_post[:,:,1,1]',vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="terrain")
			axis("off");  #colorbar(fraction=0.046, pad=0.04);
			title("Posterior samp G(z;rtm)")

			subplot(plots_len, num_cols,3); imshow(X_gt[:,:,1,1]',  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="terrain")
			axis("off"); title(L"Reference $\mathbf{x^{*}}$") ;# colorbar(fraction=0.046, pad=0.04)

			subplot(plots_len, num_cols,4); imshow(error_mean' , vmin=0, interpolation="none", cmap="magma")
			axis("off");title("RMSE "*string(rmse_i)) ; #cb = colorbar(fraction=0.046, pad=0.04)
		
			tight_layout()
			fig_name = @strdict  corr_prob n_x_tr n_y_tr noise_wells n_wells  unet_lev     n_train e lr batch_size ind
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)
		end
	
		if e != plot_every
			fig = figure(figsize=(10,5))
			subplot(3,1,1); title("Logarithm of Objective: "*string(log.(loss[end])))
			plot(range(0f0, 1f0, length=length(loss)), log.(loss), label="train");
			xlabel("Parameter Update"); legend();

			subplot(3,1,2); title("SSIM best=$(round(maximum(ssim);digits=5) ) ")
		    plot(range(0f0, 1f0, length=length(ssim)),ssim);  
		    xlabel("Parameter Update") 

		    subplot(3,1,3); title("RMSE best=$(round(minimum(l2_cm);digits=5)) ")
		    plot(range(0f0, 1f0, length=length(l2_cm)),l2_cm); 
		    xlabel("Parameter Update") 
	
			tight_layout()
			fig_name = @strdict corr_prob n_x_tr n_y_tr noise_wells n_wells  unet_lev   n_train e lr batch_size
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end
	end

	#save params every 4 epochs
    if(mod(e,save_every)==0) 
    	 #Saving parameters and logs
     	unet_cpu = unet |> cpu;
		save_dict = @strdict  unet_lev unet_cpu nx  n_train e  lr n_x_tr n_y_tr  loss  l2_cm ssim loss_test  l2_cm_test ssim_test batch_size; 
		@tagsave(
			joinpath(save_net_path, savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
		);
    end
end




 