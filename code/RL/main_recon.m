clear;

%% system parameters
gpuDevice(1);
angle=13;
system_tag = 'midNA';
disp(system_tag)
wdf_dir_root = '../../demo_data/';
save_tag = 'RL';
input_tag = 'input';
maxIter = 10;
for u=1:angle
    load(['../../psf/file_',num2str(u, '%05d'),'.mat'],'psf');
    psf1(:,:,:,u)=psf;
end
psf=psf1;
upweight = 0.5;

psf=psf./sum(sum(psf,1),2);
    

%% load wdf and reconstruct
wdf_dir_this = wdf_dir_root;

input_folder = sprintf('%s/%s', wdf_dir_this, input_tag);
disp(input_folder);
tif_files = dir(fullfile(input_folder, '*.tif'));
frame_list = [1:length(tif_files)];
save_dir_root = sprintf('%s/test/%s', wdf_dir_this, save_tag);
if ~exist(save_dir_root, 'dir')
    mkdir(save_dir_root);
end

save_dir_mip = sprintf('%s/mip_z', save_dir_root);
if ~exist(save_dir_mip, 'dir')
    mkdir(save_dir_mip);
end

for file_id=frame_list
    fprintf('recon frame %d\n', file_id);
    clear wdf;
    wdf_path = sprintf('%s/inp_%d.tif', input_folder,  file_id);
    for u=1:angle
        wdf(:,:,u)=single(imread(wdf_path,u));
    end
    wdf = double(wdf);
    wdf(wdf<1e-3) = 1e-3;
    fprintf('wdf min %.2f, max %.2f\n', min(wdf(:)), max(wdf(:)));
    Xguess=ones(size(wdf,1),size(wdf,2),size(psf,3));
    Xguess=Xguess./sum(Xguess(:)).*sum(wdf(:))./(size(wdf,3)*size(wdf,4));
    savestep=20;
    savename=false;
    Xguess = deconvRL_2p(Xguess, wdf, psf, maxIter, save_dir_root, upweight, savestep,savename);
    Xguess = gather(Xguess);
    fprintf('reconed volume min %.2f, max %.2f\n', min(Xguess(:)), max(Xguess(:)));
    volume = Xguess;
    save_path = sprintf('%s/frame_%d.tif', save_dir_root, file_id);
    imwriteTFSK(uint16(volume), save_path);
    Xguess_mip = Xguess(:, :, 6: end-5);
    disp(size(Xguess_mip))
    volume_mip = uint16(squeeze(max(Xguess_mip, [], 3)));
    save_path = sprintf('%s/recon_frame_%d.tif', save_dir_mip, file_id);
    imwriteTFSK(volume_mip, save_path);
end