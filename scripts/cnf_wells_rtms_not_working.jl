#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=50G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=50G srun --pty julia

using DrWatson
@quickactivate "GenerativeGeostats.jl"
import Pkg; Pkg.instantiate()

using PyPlot,SlimPlotting
using InvertibleNetworks, Flux, UNet
using LinearAlgebra,Random,Statistics
using ImageQualityIndexes
using DrWatson
using BSON,JLD2
using Zygote

function get_batch_well(X; width_well = 3, n_wells = 1,min_dist = 20 )
	Y = zeros(Float32,size(X))
	Y_mask = zeros(Float32,size(X))
	Y_cond = zeros(Float32,size(X))
	nx = size(X)[1]
	for b in 1:size(X)[end]
		set_possible = collect(1:(nx-width_well+1))
		wells = []
		for i in 1:n_wells
			sel_rand = rand(set_possible)
			append!(wells,sel_rand:(sel_rand+width_well-1))
			deleteat!(set_possible, findall(x->x in (sel_rand-min_dist):(sel_rand+min_dist+width_well-1),set_possible))
		end
		Y[wells,:,:,b] .= X[wells,:,:,b]
	end
	Y
end

function z_shape_simple(G, ZX_test)
    Z_save, ZX = split_states(ZX_test[:], G.cond_net.Z_dims)
    for i=G.cond_net.L:-1:1
        if i < G.cond_net.L
            ZX = tensor_cat(ZX, Z_save[i])
        end
        ZX = G.cond_net.squeezer.inverse(ZX) 
    end
    ZX
end

function z_shape_simple_forward(G, X)
	orig_shape = size(X)
    G.cond_net.split_scales && (Z_save = array_of_array(X, G.cond_net.L-1))
    for i=1:G.cond_net.L
        (G.cond_net.split_scales) && (X = G.cond_net.squeezer.forward(X))
        if G.cond_net.split_scales && i < G.cond_net.L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.cond_net.Z_dims[i] = collect(size(Z))
        end
    end
    G.cond_net.split_scales && (X = cat_states(Z_save, X))
    X = reshape(X,orig_shape)
    return X
end

#function get_cm_l2_ssim(G, Xs_batch, Y_batch, Z_batch; device=gpu, num_samples=1)
function get_cm_l2_ssim(G, Xs_batch, Y_batch; device=gpu, num_samples=1)
		# needs to be towards target so that it generalizes accross iteration
	    num_test = size(Y_batch)[end]
	    l2_total = 0 
	    ssim_total = 0 
	    #get cm for each element in batch
	    for i in 1:num_test
	    	Z = randn(Float32, nx,ny,1,num_samples)
	    	#Z = Z_batch[:, :, :, i:i];
			conds = Y_batch[:, :, :, i:i];
			x_gt  = Xs_batch[:,:,1,i] |> cpu
			
			conds = repeat(conds |>cpu, 1, 1, 1, num_samples) 
			X_gen_vec, Zy = G.forward(Z |> device, conds|> device )[1:2]
			X_post = z_shape_simple(G,X_gen_vec) |> cpu

	    	x_hat =   mean(X_post; dims=4)[:,:,1,1]
	    	ssim_total += assess_ssim(x_hat, x_gt)
			l2_total   += sqrt(mean((x_hat - x_gt).^2))
		end
	return l2_total / num_test, ssim_total / num_test
end

# Plotting configs
save_net_path = "/slimdata/rafaeldata/savednets_krig_hint2"
plot_path = "/slimdata/rafaeldata/plots/unsuper_krig/cnf-wells-rtm"

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)

# Training hyperparameters 
device = gpu

lr      = 1f-5
clipnorm_val = 3f0
noise_lev_x = 0.000f0
noise_lev_y = 0.0f0 #

batch_size   = 8#
n_epochs     = 200
num_post_samples = 32

save_every   = n_epochs
plot_every   = 2
n_condmean   = batch_size


data_path = "/slimdata/rafaeldata/fwiuq_eod/rtms_oed.jld2"
m_train = JLD2.jldopen(data_path, "r")["m_train"];
grad_train = JLD2.jldopen(data_path, "r")["grad_train"];
grad_train = mean(grad_train;dims=3)


nx = 256
m_train    = m_train[128:383,:,:,:]
grad_train = grad_train[128:383,:,:,:]

max_y = quantile(abs.(vec(grad_train[:,:,:,1:300])),0.9999);
grad_train ./= max_y;

Random.seed!(123);
Xs = m_train
Ys = grad_train

n_total = size(Xs)[end]
n_in = 1

vmax_v = maximum(Xs)
vmin_v = minimum(Xs)

# training indexes 
i_train = 800

Random.seed!(122);
n_wells = 2
width_well = 3
Xs_wells = get_batch_well(Xs;width_well = width_well, n_wells=n_wells,min_dist = 0 )
Ys_wells = Ys

X_train  = Xs_wells[:,:,:,1:i_train]
Y_train  = Ys_wells[:,:,:,1:i_train]
println(maximum(X_train[1,:,:,:])) #there needs to be at least one selected in first and last column
println(maximum(X_train[end,:,:,:])) #there needs to be at least one selected in first and last column

i_train_test = 800
Xs_test   = Xs[:,:,:,(i_train_test+1):end]
X_test   = Xs_wells[:,:,:,(i_train_test+1):end]
Y_test   = Ys_wells[:,:,:,(i_train_test+1):end]

n_c  = size(Y_train)[end-1] 

nx, ny, _, n_train = size(X_train)
N = nx*ny

n_batches      = cld(n_train, batch_size)-1
n_train        = batch_size*n_batches

# need to restart from here!
# Architecture parametrs
sum_net = true
unet_lev = 4
unet = Chain(Unet(n_c,n_in,unet_lev))
trainmode!(unet, true)  
unet = FluxBlock(unet) |> device


# Create conditional network
L = 3
K = 9
n_hidden = 64
low = 0.5f0
logdet = true
rand_aug = true
first_actnorm = true
har = true
Random.seed!(123);
cond_net = NetworkConditionalGlow(1, n_in, n_hidden,  L, K; squeezer=HaarLayer(),split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

# Optimizer
opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss   = []; logdet_train = []; ssim   = []; l2_cm  = [];

loss_test   = []; logdet_test = []; ssim_test   = []; l2_cm_test  = [];

#factor = Float32.(sqrt(9/(320*320)))
factor = 0.01f0

prior_mean = mean(Xs)
prior_std = std(Xs)


z_fix = false
for e=1:n_epochs# epoch loop
	idx_e = reshape(randperm(n_train), batch_size, n_batches) 
    for b = 1:n_batches # batch loop
    	@time begin
    	Z = randn(Float32, nx,ny,1,batch_size) 	
		conds = Y_train[:, :, :, idx_e[:,b]]
		Y     = X_train[:, :, :, idx_e[:,b]] 
		Y += noise_lev_x*randn(Float32,size(Y))

		#flip augmentation
		for i in 1:batch_size
		  	if rand() > 0.5
	    		conds[:,:,:,i:i] = conds[end:-1:1,:,:,i:i]
	    		Y[:,:,:,i:i] = Y[end:-1:1,:,:,i:i]
		  	end
	    end
		Y = Y |> device

		X_gen_vec, Zy, lgdet = G.forward(Z |> device, conds |> device;)
	    X_gen = z_shape_simple(G,X_gen_vec)

	    mask = Y .> 0
	    res = (mask.*X_gen .- Y) 
	    total_grad = z_shape_simple_forward(G,res)

	    # Set gradients of flow and summary network
	    dx, x, dy = G.backward((total_grad / factor) / batch_size, X_gen_vec, Zy; Y_save = conds |> device)

	     # Loss function is l2 norm 
	    append!(loss, norm(res)^2 / (N*batch_size))
	    append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

	    for p in get_params(G) 
	      Flux.update!(opt,p.data,p.grad)
	    end; clear_grad!(G)

	    print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	        "; f l2 = ",  loss[end], 
	        "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
	    Base.flush(Base.stdout)
		end
    end

    if(mod(e,plot_every)==0) 

    	# get conditional mean metrics over training batch  
	    @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(G, Xs[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
	    append!(ssim, cm_ssim_train)
	    append!(l2_cm, cm_l2_train)

	    # get conditional mean metrics over testing batch  
	    @time cm_l2_test, cm_ssim_test  = get_cm_l2_ssim(G, Xs_test[:,:,:,1:n_condmean], Y_test[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
	    append!(ssim_test, cm_ssim_test)
	    append!(l2_cm_test, cm_l2_test)

	    #testmode!(unet, true)
		num_cols = 4
	    plots_len = 2
	    inds = [1,2] 
	    for i in 1:plots_len
	    	ind = inds[i]
	    	
		    Z  = randn(Float32, nx,ny,1,num_post_samples)
			conds = Y_train[:, :, :, ind:ind];

			X_gt  = Xs[:,:,:,ind:ind] |> cpu
			Y     = X_train[:,:,:,ind:ind] |> cpu

			conds = repeat(conds |>cpu, 1, 1, 1, num_post_samples) 
			X_gen_vec, Zy = G.forward(Z |> device, conds|> device )[1:2]
			X_post = z_shape_simple(G,X_gen_vec) |> cpu

			X_post_mean = mean(X_post,dims=4)
			X_post_std  = std(X_post, dims=4)

			x_hat = X_post_mean[:,:,1,1]	

			mask = Y .> 0 
	    	res = (mask.*x_hat - Y)

			error_mean = abs.(x_hat-X_gt[:,:,1,1])
			ssim_i = round(assess_ssim(x_hat, X_gt[:,:,1,1]),digits=2)
			rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)

			y_plot = conds[:,:,1,1]' |> cpu
			a = quantile(abs.(vec(y_plot)), 98/100)

			fig = figure(figsize=(11, 5)); 
			subplot(plots_len, num_cols,1); imshow(y_plot, vmin=-a,vmax=a,interpolation="none", cmap="gray")
			axis("off"); title("rtm");#colorbar(fraction=0.046, pad=0.04);

			subplot(plots_len, num_cols,2); imshow(Y[:,:,1,1]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
			axis("off"); title(L"$\mathbf{y} = \mathbf{F}(\mathbf{x^{*}})$ observation"); #colorbar(fraction=0.046, pad=0.04);

			subplot(plots_len, num_cols,3); imshow(X_post[:,:,1,1]',vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
			axis("off");  #colorbar(fraction=0.046, pad=0.04);
			title("Posterior samp G(z;rtm)")

			subplot(plots_len, num_cols,4); imshow(X_post[:,:,1,end]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
			axis("off");  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")
			title("Posterior samp G(z;rtm)")

			subplot(plots_len, num_cols,5); imshow(X_gt[:,:,1,1]',  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="cet_rainbow4")
			axis("off"); title(L"Reference $\mathbf{x^{*}}$") ;# colorbar(fraction=0.046, pad=0.04)

			subplot(plots_len, num_cols,6); imshow(x_hat' ,  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="cet_rainbow4")
			axis("off"); title("Posterior mean SSIM="*string(ssim_i)) ; #colorbar(fraction=0.046, pad=0.04)

			subplot(plots_len, num_cols,7); imshow(error_mean' , vmin=0,vmax=1, interpolation="none", cmap="magma")
			axis("off");title("RMSE "*string(rmse_i)) ; cb = colorbar(fraction=0.046, pad=0.04)

			subplot(plots_len, num_cols,8); imshow(X_post_std[:,:,1,1]' , vmin=0,vmax=1,interpolation="none", cmap="magma")
			axis("off"); title("Posterior variance") ;cb =colorbar(fraction=0.046, pad=0.04)

			tight_layout()
			fig_name = @strdict har rand_aug  z_fix  n_wells  factor unet_lev   clipnorm_val  n_train e lr n_hidden L K batch_size ind
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)
		end

		if e != plot_every
			sum_train = loss + logdet_train 

			fig = figure("training logs ", figsize=(10,12))
			subplot(6,1,1); title("L2 Term: train="*string(loss[end]))
			plot(range(0f0, 1f0, length=length(loss)), loss, label="train");
			xlabel("Parameter Update"); legend();

			subplot(6,1,2); title("Logdet Term: train="*string(logdet_train[end])*" test=")
			plot(range(0f0, 1f0, length=length(logdet_train)),logdet_train);
			xlabel("Parameter Update") ;

			subplot(6,1,4); title("Total Objective: train="*string(sum_train[end])*" test=")
			plot(range(0f0, 1f0, length=length(sum_train)),sum_train); 

			subplot(6,1,5); title("SSIM train=$(ssim[end]) test=$(ssim_test[end])")
		    plot(range(0f0, 1f0, length=length(ssim)),ssim); 
		    plot(range(0f0, 1f0, length=length(ssim_test)),ssim_test); 
		    xlabel("Parameter Update") 

		    subplot(6,1,6); title("RMSE train=$(l2_cm[end]) test=$(l2_cm_test[end])")
		    plot(range(0f0, 1f0, length=length(l2_cm)),l2_cm); 
		    plot(range(0f0, 1f0, length=length(l2_cm_test)),l2_cm_test); 
		    xlabel("Parameter Update") 
	
			tight_layout()
			fig_name = @strdict har rand_aug  z_fix  n_wells  factor unet_lev   clipnorm_val  n_train e lr n_hidden L K batch_size
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end
	end

    if(mod(e,save_every)==0) 
     	unet_model = G.sum_net.model|> cpu;
        G_save = deepcopy(G);
        reset!(G_save.sum_net); # clear params to not save twice
		Params = get_params(G_save) |> cpu;
		save_dict = @strdict har rand_aug nx   logdet factor  unet_lev unet_model  nx  clipnorm_val n_train e noise_lev_x noise_lev_y lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size; 
		@tagsave(joinpath(save_net_path, savename(save_dict, "bson"; digits=6)),save_dict;safe=true);
    end
end




 