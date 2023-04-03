%% use L_4 to estimate the overcomplete blind deconvolution problem || A^ diag(h) Y||_4^4
% 12/18/2019


% clear all;clc;



%% experiment parameters setting

% % the great formulation now for  m fixed
% n_all=[16,32,64]; % dimension of observation y_i
% K_all=[8,4,2];  % number of the filter
% m_all=128; % the dimension of x_i
% m=128;
% p_all=m^2;

% test for p
% n_all=[64,32,16]; % dimension of observation y_i
n_all = [64];
K_all=4;  % compressing ratio
m_all=K_all*n_all; % the dimension of x_i
% p_all=[1536,1024,512,256];  % the number of measurements
p_all = [2048];

% kappa is the ratio of max|g|/min|g| 
kappa_all=[2];
errors=[];

orthogonal_ornot=0; % whether the circulated matrix is orthogonal
% theta_all =[.1,.3,.5];   % sparsity level of X
theta_all = [0.3];


%% algorithm parameters
max_iter = 400; % neighborhood size (stopping criterion)
critical_stop_min_iter=30;
success_eps = 1-0.01; 
difference_eps=10^-6;
%% stopping critera
max_grad_thre=1e-2;

% parameter for optimization algorithm 
lbeta=0.8;
alpha=0.9;


%% 
% success_vec = zeros(length(n_all),length(theta_all));  % under dimension, change sample complexity
success_vec = zeros(length(n_all),length(p_all),length(theta_all));  % under dimension
    
for ni=1:length(n_all) % the dimension of the observation
    n=n_all(ni);
%     K=K_all(ni);
    m=m_all(ni)
    for oi=1:length(kappa_all)       % for conditional parameter
        kappa=kappa_all(oi);

for nm = 1:length(p_all)            % number of measurements
	p = p_all(nm);    

for k = 1:length(theta_all)            % different sparsity level theta
	theta = theta_all(k);    
	success = zeros(10, 1);
    
    
    for j = 1:10  % 10 random instances
        
        %% start the experiment for one scenario

		all_R = zeros(n,m);     % the circulated of filter matrix
        all_normed_R=zeros(n,m);
        % get the not orthogonal circulated matrix R
        F=dftmtx(m)/sqrt(m);
        FH=conj(F);
        
        % test which A is better
        ratio = 4;
        F_wide = F(find(mod(1:m,ratio)==3),:);
        atest = F_wide'*F_wide;
        atest(logical(eye(size(atest)))) = 0;
        
        
        ground_truth_g=zeros(n,1);
        dft_gain_f=unifrnd(0,kappa,n,1);
        dft_phase_f=unifrnd(0,2*pi,n,1);
        ground_truth_g=dft_gain_f.*exp(1i*dft_phase_f);     
        diag_g=diag(ground_truth_g);
        
%         F_wide=F(find(mod(1:m,K)==1),:);
%         F_wide=F(1:n,:);
        random_select_row=randperm(m);
        F_wide=F(random_select_row(1:n),:);
%         P = eye(m);
%         P = P(randperm(m),:);   
%         F_wide=F(randperm(m,n),:)*P;  
        

        A=F_wide;
        
        % change A to UNTF
%         A=GetUNTF(n,m)*sqrt(n/m);
        
%         A = (randn(n, m)+1i*randn(n, m))/sqrt(2*m);
        %%
        all_R=diag_g*A;
        for ri=1:m
            all_normed_R(:,ri)=all_R(:,ri)/norm(all_R(:,ri));
        end
        
		X = (1/2*randn(m,p)+1/2i*randn(m,p)).*(rand(m, p) <= theta);   % iid Bern-Gaussian model
        
% % %         get a not good X
%         x_mag = unifrnd(0,1000,m,1);
%         
%         X = diag(x_mag)*randn(m,p).*(rand(m, p) <= theta);
        Y=all_R*X;
        
        
        %% calcualte the preconditioner
        pre_R=zeros(n,n);
        fft_yi=zeros(n,1);
        for pre_i=1:p
            pre_i
            pre_R=pre_R+diag(Y(:,pre_i))*A*(diag(Y(:,pre_i))*A)'/(theta*m*p);
        end
            pre_R=inv(pre_R)^(1/2);
        disp('finished pre_R')
               
		for l = 1:10  % for 10 times of random initialization
            % for test whether it converge, or just random arrive 0.99
            flag_converge=0;
            %
			disp(['k = ', num2str(k), ', j = ', num2str(j), ', l = ', num2str(l)]);
            q=1/2*randn(n,1)+1/2i*randn(n,1);
			q = q/norm(q);
            % test why can't be the stationary point
            judge_recover_old2=0;
            judge_recover_old=0;
    
            for i = 1:max_iter
                

                eta=0.1;
                % calculate the subgradient
                egrad=zeros(n,1);
      
%                 tic
%                 %% calculate gradient
%                 for gi=1:p
%                     objec_pre= A'* diag(pre_R*Y(:,gi)) * q;
%                     egrad=egrad - 1/2/p* conj( diag(pre_R*Y(:,gi) ) *  conj(A) * diag( conj(objec_pre).^2 ) * objec_pre );
%                 end
%                 toc
                tic
                % version 2 for calculate gradient
                prey = pre_R*Y;
                objec_preall = A' * diag(q) * prey;
                objec3A = conj(A) * (conj(objec_preall).^2 .* objec_preall);
                egrad = -1/2/p*sum(conj(prey.* objec3A), 2);

                toc
%                 tic
%                 xLayer=n-1;
%                 q_after_pre_R=vec2mat(pre_R*q,n);
%                 for gi=1:p
%                 y1=vec2mat( Y(:,gi),n);
%                 y2=circshift( fliplr(flipud(y1)), [1,1] );
% %               y2=fliplr(y2);
% %               y2=circshift(y2,[1,1]);
%                 B_y1=padarray(y1,[xLayer,xLayer],'circular');
%                 B_y2=padarray(y2,[xLayer,xLayer],'circular');
%                 cyi=conv2(q_after_pre_R,B_y1);
%                 L3_value=cyi(xLayer+1:xLayer+n, xLayer+1:xLayer+n).^3;
%                 cyi_multi_trans=conv2(L3_value,B_y2);
%                 cyi_multi_trans=cyi_multi_trans(xLayer+1:xLayer+n, xLayer+1:xLayer+n);
%                 egrad=egrad-reshape(cyi_multi_trans',[],1);
%                 end



      %%%%%%%%%%%%%%%%%%%%%%%%%%% back searching
       
      %% back searching process to select eta
                object_old=-1/4/p* sum( sum(abs(A'* diag(q)*pre_R*Y ).^4) );
                object_new=object_old;
                while (object_new>object_old-alpha*eta*norm(egrad - q*(q'*egrad))^2 )
                    eta=eta*lbeta;
                    q_new=q - eta * (egrad - q*(q'*egrad));
                    q_new=q_new/norm(q_new);
                    object_new=-1/4/p* sum( sum(abs(A'* diag(q_new)*pre_R*Y ).^4) );
                end
        % do gradient descent     
		q = q - eta * (egrad - q*(q'*egrad));    % Riemannian step
		q = q/norm(q);   % projection
                
				 
      
      
      
        
      % early stopping if recovered
%                 % record the value of loss function
      q_final=inv(pre_R)'* diag( conj(q) )*A;
      q_final=q_final./repmat(vecnorm(q_final),n,1);
      
      judge_recover_new2=-1/4/p* sum( sum(abs(A'* diag(q)*pre_R*Y ).^4) );
      

        % the criteria for success recovery
      judge_recover_new= max(max(abs(((q_final(:, 1)'*all_normed_R)) ) ) )/norm(q_final(:, 1))
      
      angles= exp(-1i*angle((q_final(:,1)'*all_normed_R)));
      
      q_final2 = inv(pre_R)'* conj(q);
      judge_recover_new = max(max(abs(((q_final2'*all_normed_R)) ) ) )/norm(q_final2)
      
%       judge_recover_new=max(abs(real(q_final(:,1)'*all_normed_R*diag(angles)  )))
      errors=[errors,judge_recover_new];
           
                %% judge the success
       if(judge_recover_new>success_eps && flag_converge<=3)
           flag_converge=flag_converge+1;
       elseif(judge_recover_new>success_eps)
            success(j)=1;
%            break;
       end
                
                % early stopping if gradient is too small
%                 max(abs(egrad - q*(q'*egrad)));
%                 if (max(abs(egrad - q*(q'*egrad)))<max_grad_thre)
% %                      break;
%                 end
                if( abs(judge_recover_new2-judge_recover_old2)<=difference_eps && i>critical_stop_min_iter)
                    break;
                end

                judge_recover_old2=judge_recover_new2;
                toc
                
            end
            if(success(j)==1)
                break;
            end
            
        end
    
    end
    disp(['Success rate for n = ', num2str(n), ' m = ', num2str(m), ': ', num2str(mean(success))]);
	success_vec(ni,nm,k) = mean(success);
end

    end
    end
end

C=diag(q_final(:,1))*A;
e_1=ones(2*m,1);
C=A;
cvx_begin
   variable x(m) complex
   minimize( norm(x, 1)   );
   subject to
      C*x == diag(1./ground_truth_g)*Y(:,1);
cvx_end
Cx=max(abs(x'*FH* diag( fft(X(:,1)) )*F /norm(x,2)/norm(X(:,1)) ) );

figure;
for i=1:length(p_all)
    plot(p_all,success_vec(i,:,2));
    hold on
end


figure;
% phase_transition=flipud(reshape(success_vec(1,:,:),3,3))
imagesc(p_all,theta_all,reshape(success_vec(1,:,:),3,3))
colormap gray;
set(gca,'YDir','normal') 
% xlabel('n');
xlabel('\kappa');
% xlabel('\theta');
ylabel('p');