function Xguess = deconvRL_2p(Xguess, wdf, psf, maxIter, savepath, upweight, savestep, savename)


    wdf = gpuArray(single(wdf));
    Xguess=gpuArray(single(Xguess));
        
        
    % Iterative deconvolution algorithm.
    for iter=1:maxIter
        tic;
    
        % Volume update.
        for u=size(wdf,3):-1:1
            
            WDF_u=wdf(:,:,u);
            psf_u=gpuArray(single(squeeze(psf(:,:,:,u))));
            forwardFUN = @(Xguess) forwardProjectGPU( psf_u, Xguess );
            backwardFUN = @(projection) backwardProjectGPU(psf_u, projection );
            uniform_matrix = gpuArray(single (ones(  size(wdf,1),size(wdf,2) )));
            
            % RL deconvolution.
            HXguess=forwardFUN(Xguess);
            errorEM=squeeze(WDF_u)./HXguess;
            errorEM(~isfinite(errorEM))=0;
            XguessCor = backwardFUN(errorEM) ;
            Htf=backwardFUN( uniform_matrix );
            Htf(Htf<1e-4)=0;
            Xguess_add=Xguess.*XguessCor./Htf;
            clear Htf;clear XguessCor;
            Xguess_add(find(isnan(Xguess_add))) = 0;
            Xguess_add(find(isinf(Xguess_add))) = 0;
            Xguess_add(Xguess_add<0 ) = 0;
            Xguess=Xguess_add.*upweight+(1-upweight).*Xguess;
            clear Xguess_add;
            Xguess(find(isnan(Xguess))) = 0;
            Xguess(Xguess<1e-4)=1e-4;
            
        end
        ttime = toc;
        disp(['  iter ' num2str(iter) ' | ' num2str(maxIter),', took ' num2str(ttime) ' secs']); 
        if ~isa(savepath, 'logical') && mod(iter, savestep) == 0
            imwriteTFSK(uint16(gather(Xguess)),[savepath,'/', sprintf('%s', savename), '_iter',num2str(iter),'.tif']);
        end
    end
    
