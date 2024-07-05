clear variables; close all; clc;

%% Define constants
kb_source = 1;  % For simplicity, taking kb = 1
lambda_source = 2 * pi / 1;  % Wavelength of the background field

%% Setup Configuration
Nx=20;
Ny=Nx;
h = lambda_source / 20;  % Step size for the grid

% Define the corners of the rectangle domain D
x0 = 0; y0 = 0;  % Upper left corner at the origin
x1 = lambda_source; y1 = lambda_source;  % Lower right corner at (λ, λ)

% Create the grid
x_values = x0+h/2:h:x1;
y_values = y0+h/2:h:y1;
[X, Y] = meshgrid(x_values, y_values);

N=numel(X,Y);
display(['Number of points in the grid: ', num2str(N)]);
figure(1);

% Plot the rectangle domain D
line([x0, x1], [y0, y0], 'Color', 'b');  % Top edge
line([x1, x1], [y0, y1], 'Color', 'b');  % Right edge
line([x1, x0], [y1, y1], 'Color', 'b');  % Bottom edge
line([x0, x0], [y1, y0], 'Color', 'b');  % Left edge

% Plot the grid points
hold on;
plot(X, Y, 'k.', 'MarkerSize', 5);

% Plot the source location
rho_s_x = lambda_source / 2;
rho_s_y = 10 * lambda_source;

plot(rho_s_x, rho_s_y, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
set(gca, 'YDir','reverse')
text(rho_s_x, rho_s_y, '  Source', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% Set axis limits and labels
axis equal;
xlim([-2*lambda_source-1, 2.5*lambda_source+1])
ylim([-1 1.3*rho_s_y])
xlabel('x');
ylabel('y');
title('Source Location and Imaging Domain  ');
grid on;
saveas(gcf,'source_object_domain','epsc');

%% Incident field computation
% Compute the distance from each grid point to the source
R = sqrt((X - rho_s_x).^2 + (Y - rho_s_y).^2);

kb=100*kb_source;

% Compute the incident field
uinc = -1j / 4 * besselh(0, 2, kb * R);

% PLOT COMPONENTS
%  figure;
% t=tiledlayout(1,3);
% t.TileSpacing='tight';
% 
% nexttile;
% imagesc(x_values, y_values, real(uinc));
% C = colorbar('location','southoutside');
% axis equal tight;
% title('Real Part of $\hat{u}_{inc}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile;
% imagesc(x_values, y_values, imag(uinc));
% C = colorbar('location','southoutside');
% axis equal tight;
% title('Imaginary Part of $\hat{u}_{inc}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile;
% imagesc(x_values, y_values, abs(uinc));
% C = colorbar('location','southoutside');
% axis equal tight;
% title('Absolute Value of $\hat{u}_{inc}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% outerPos = t.OuterPosition;
% outerPos(2) = outerPos(2) + .25; % Adjust the bottom position
% outerPos(4) = outerPos(4) - .35; % Adjust the height
% 
% t.OuterPosition = outerPos;
% 
% tt=title(t,'Incident field components');
% saveas(gcf,'u_inc_farfield','epsc');

% %% Move source closer and compute again the incident field
% 
% rho_s_x_2=lambda_source/2;
% rho_s_y_2=2*lambda_source;
% 
% % Compute the distance from each grid point to the source
% R = sqrt((X - rho_s_x_2).^2 + (Y - rho_s_y_2).^2);
% 
% % Compute the incident field
% uinc_close = -1j / 4 * besselh(0, 2, kb * R);
% 
% figure;
% t=tiledlayout(1,3);
% t.TileSpacing='tight';
% 
% nexttile;
% imagesc(x_values, y_values, real(uinc_close));
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Real Part of $\hat{u}_{inc}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile;
% imagesc(x_values, y_values, imag(uinc_close));
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Imaginary Part of $\hat{u}_{inc}$','Interpreter','latex');
% xlabel('x'); ylabel('y');
% 
% nexttile;
% imagesc(x_values, y_values, abs(uinc_close));
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Absolute Value of $\hat{u}_{inc}$','Interpreter','latex');
% xlabel('x'); ylabel('y');
% 
% outerPos = t.OuterPosition;
% outerPos(2) = outerPos(2) + .25; % Adjust the bottom position
% outerPos(4) = outerPos(4) - .35; % Adjust the height
% t.OuterPosition = outerPos;
% 
% tt=title(t,'Incident field components - Closer source');
% saveas(gcf,'u_inc nearfield','epsc')

%% Add object
image=imread('mug.jpg');
if size(image,3)==3
    image=rgb2gray(image);
end

image=imresize(image,[length(y_values),length(x_values)]);
chi=double(image)/255;
chi=abs(1-chi);
chi=chi*100+eps;

figure;
imagesc(chi);
colormap(flipud('Gray'))
colorbar;

title('Contrast function $\chi$','Interpreter','latex');
saveas(gcf,'contrasfunc','epsc');

 %% Receiver Domain

% Define the end points of the receiver domain Drec
x_start = -lambda_source;
y_start = 1.5 * lambda_source;
x_end = 2 * lambda_source;
y_end = 1.5 * lambda_source;

figure(1); %add to original setup
line([x_start, x_end], [y_start, y_end], 'Color', 'r', 'LineWidth', 2);
set(gca, 'YDir','reverse')
hold on;

M=400; % # of receivers

% Compute the step size based on M
t = linspace(0, 1, M);
x_rec = x_start + t * (x_end - x_start);
y_rec = y_start + t * (y_end - y_start);

% Plot the receiver domain Drec as a line segment with grid points
plot(x_rec, y_rec, 'ro-', 'LineWidth', 1, 'MarkerSize', 4);
title('Receivers setup');
axis equal tight;
xlim([-2*lambda_source-1, 2.5*lambda_source+1])
ylim([0 1.3*rho_s_y])
hold on;

saveas(gcf,'setup_line','epsc');

%%%  compute system matrix

A=zeros(M,N);
receivers_locs=[x_rec; y_rec];

G=@(r) -1j/4 * besselh(0,2,r); % define green's func

for i=1:M
    dist_prime = sqrt((receivers_locs(1,i) - X).^2 + (receivers_locs(2,i)-Y).^2);
    for m=1:Ny
        for n=1:Nx
            a_temp(m,n)=G(kb*dist_prime(m,n))*uinc(m,n);
        end
    end
    A(i,:)=reshape(a_temp,1,[]);  


end

chi_flat=reshape(chi,[size(chi,1)*size(chi,2) 1]);

uscat=A*chi_flat.*kb^2;

%FOR THE REPORT

% [U, S, V] = svd(A, 'vector');
% figure;
% semilogy(S);
% xlabel('Singular Value Index')
% ylabel('Singular Value')
% title(sprintf('Singular Values of A - %d receivers ',M));
% saveas(gcf,sprintf('singular_values_M%d',M),'epsc');

%% Compute the scattered field element wise -- sanity check
% uscat_2=zeros(M,1);
% 
% for i=1:M
%    dist_prime = sqrt((receivers_locs(1,i) - X).^2 + (receivers_locs(2,i)-Y).^2);
%     for m=1:Ny
%         for n=1:Nx
%             green_term=(-j/4)*besselh(0,2,kb*dist_prime(m,n));
%             uscat_2(i)=uscat_2(i)+green_term*chi(m,n)*uinc(m,n);
%         end
%     end
%     uscat_2(i)=uscat_2(i)*kb^2;
% end
%% Contrast Reconstructions - Noiseless Cases
figure;

% Pinv from SVD
[U, S, V] = svd(A, 'econ', 'vector');
S_inv = zeros(size(S));
non_zero_entries = S > eps; %avoid close to 0 singular values, below eps there's no meaning
S_inv(non_zero_entries) = 1 ./ S(non_zero_entries);

xmn_svd=V * diag(S_inv) * U' * uscat;
xmn_svd=reshape(xmn_svd,[Ny Nx]);

imagesc(x_values, y_values, abs(xmn_svd));
colormap(flipud('Gray'))
C = colorbar('location','southoutside');
axis equal tight;
titlestr=sprintf('${x}_{mn} M=%d k_b=%d$',M,kb);
title(titlestr,'Interpreter','latex');
xlabel('x');
ylabel('y');
%saveas(gcf, sprintf('line_chi_recon_m%d_kb%d',M, kb),'epsc');
%% Noisy cases
SNR_dB = 0;

signal_power = mean(abs(uscat).^2);
noise_power = signal_power / (10^(SNR_dB / 10));
noise_std = sqrt(noise_power);
noise = noise_std * (randn(M, 1) + 1i * randn(M, 1));

uscat_noisy = uscat + noise;

% Pinv from SVD
[U, S, V] = svd(A,'econ','matrix');
S=diag(S);
S_inv = zeros(size(S));
non_zero_entries = S > eps; %avoid close to 0 singular values, below eps there's no meaning
S_inv(non_zero_entries) = 1 ./ S(non_zero_entries);

xmn_svd_noisy= V * diag(S_inv) * U' * uscat_noisy;
xmn_svd_noisy=reshape(xmn_svd_noisy,[Ny Nx]);

figure;
t=tiledlayout(1,2);
nexttile; semilogy(S); subtitle('S');
nexttile; semilogy(S_inv); subtitle('S_inv')


figure;
t=tiledlayout(1,2);
nexttile;
imagesc(x_values, y_values, abs(xmn_svd_noisy));
colormap(flipud('Gray')); C = colorbar('location','southoutside');
axis equal tight;
titlestr=sprintf('${x}_{mn} M=%d k_b=%d SNR=%d$',M,kb,SNR_dB);
title(titlestr,'Interpreter','latex');
xlabel('x');
ylabel('y');

truncSVD = @(U,S,V,p) V(:,1:p)*diag(S(1:p))*U(:,1:p)';

k=75;
xmn_trunc_svd_noisy=truncSVD(U,S_inv,V,k)*uscat_noisy;
xmn_trunc_svd_noisy=reshape(xmn_trunc_svd_noisy,[Ny Nx]);

nexttile;
imagesc(x_values, y_values, abs(xmn_trunc_svd_noisy));
colormap(flipud('Gray')); C = colorbar('location','southoutside');
axis equal tight;
titlestr=sprintf('${x}_{mn} M=%d k_b=%d SNR=%d$, z=%d',M,kb,SNR_dB,k);
title(titlestr,'Interpreter','latex');
xlabel('x');
ylabel('y');

saveas(gcf,sprintf('line_noisy_M%d_kb%d_SNR=%d$',M,kb,SNR_dB), 'epsc');

% %%%%%% SQUARE
% 
% % Center the square around the object domain
% center_x = (x0 + x1) / 2;
% center_y = (y0 + y1) / 2;
% lambda=lambda_source;
% % Define the corners of the square domain Drec
% square_half_side = 1.5 * lambda_source;
% x_square_start = center_x - square_half_side;
% y_square_start = center_y - square_half_side;
% x_square_end = center_x + square_half_side;
% y_square_end = center_y + square_half_side;
% 
% % Create a figure
% figure(1);
% line([x_square_start, x_square_end, x_square_end, x_square_start, x_square_start], ...
%      [y_square_start, y_square_start, y_square_end, y_square_end, y_square_start], ...
%      'Color', 'r', 'LineWidth', 2);
% set(gca, 'YDir', 'reverse')
% hold on;
% 
% % Define the receiver positions along the edges of the square
% x_rec = [];
% y_rec = [];
% 
% % Number of receivers per side (approximate)
% M_side = 40;
% M =M_side*4;
% 
% % Top side
% x_rec = [x_rec, linspace(x_square_start, x_square_end, M_side)];
% y_rec = [y_rec, repmat(y_square_start, 1, M_side)];
% 
% % Right side
% x_rec = [x_rec, repmat(x_square_end, 1, M_side)];
% y_rec = [y_rec, linspace(y_square_start, y_square_end, M_side)];
% 
% % Bottom side
% x_rec = [x_rec, linspace(x_square_end, x_square_start, M_side)];
% y_rec = [y_rec, repmat(y_square_end, 1, M_side)];
% 
% % Left side
% x_rec = [x_rec, repmat(x_square_start, 1, M_side)];
% y_rec = [y_rec, linspace(y_square_end, y_square_start, M_side)];
% 
% % Plot the receiver positions
% plot(x_rec, y_rec, 'ro-', 'LineWidth', 1, 'MarkerSize', 4);
% axis equal tight;
% xlim([-2*lambda, 3*lambda])
% ylim([-2*lambda, 1.3*rho_s_y])
% title('Receivers in a Square Centered Around the Object Domain')
% xlabel('x-axis (horizontal)');
% ylabel('y-axis (vertical)');
% hold off;
% 
% saveas(gcf,'setup_square','epsc')
%  
% %%%
% lambda=2*pi/kb;
% freq=c/lambda;
% 
% A=zeros(M,N);
% receivers_locs=[x_rec; y_rec];
% 
% G=@(r) -1j/4 * besselh(0,2,r); % define green's func
% 
% for i=1:M
%     dist_prime = sqrt((receivers_locs(1,i) - X).^2 + (receivers_locs(2,i)-Y).^2);
%     for m=1:Ny
%         for n=1:Nx
%             a_temp(m,n)=G(kb*dist_prime(m,n))*uinc(m,n);
%         end
%     end
%     A(i,:)=reshape(a_temp,1,[]);  
% end
% 
% chi_flat=reshape(chi,[size(chi,1)*size(chi,2) 1]);
% 
% uscat=A*chi_flat.*kb^2;
% 
% %%% Contrast Reconstructions - Noiseless Cases
% xmn=pinv(A)*uscat;
% xmn=reshape(xmn,Ny,Nx);
% 
% figure;
% t=tiledlayout(1,3);
% t.TileSpacing='tight';
% title(t,'$\chi$ Reconstructions - Noiseless Cases','Interpreter','latex');
% 
% nexttile; %% left original chi
% imagesc(x_values, y_values, chi); 
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Original contrast ${\chi}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile; % middle using pinv
% imagesc(x_values, y_values, abs(xmn));
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - Pinv','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% % Pinv from SVD
% [U, S, V] = svd(A, 'econ', 'vector');
% S_inv = 1 ./(S+eps); %add eps to avoid nans or infs 
% xmn_svd = V * diag(S_inv) * U' * uscat;
% xmn_svd=reshape(xmn_svd,Ny,Nx);
% 
% nexttile;
% imagesc(x_values, y_values, abs(xmn_svd));
% colormap(flipud('Gray'))
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - SVD based','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% outerPos = t.OuterPosition;
% outerPos(2) = outerPos(2) + .25; % Adjust the bottom position
% outerPos(4) = outerPos(4) - .35; % Adjust the height
% t.OuterPosition = outerPos;
% 
% saveas(gcf,'noiseless_reconstructions_square_setup', 'epsc');
% 
% %%% Noisy cases
% noise_std=1;
% uscat_noisy=uscat+randn(M,1)*noise_std;
% 
% xmn_noisy=pinv(A)*uscat_noisy;
% xmn_noisy=reshape(xmn_noisy,[Ny Nx]);
% 
% % Pinv from SVD
% [U, S, V] = svd(A, 'econ', 'vector');
% S_inv = 1 ./(S+eps); %add eps to avoid nans or infs 
% 
% xmn_svd_noisy= V * diag(S_inv) * U' * uscat;
% xmn_svd_noisy=reshape(xmn_svd_noisy,[Ny Nx]);
% 
% truncSVD = @(U,S,V,p) V(:,1:p)*diag(S(1:p))*U(:,1:p)';
% 
% k=130;
% xmn_trunc_svd_noisy=truncSVD(U,S_inv,V,k)*uscat;
% xmn_trunc_svd_noisy=reshape(xmn_trunc_svd_noisy,[Ny Nx]);
% 
% figure(33)
% t=tiledlayout(1,2);
% nexttile; semilogy(S); subtitle('S');
% nexttile; semilogy(S_inv); subtitle('S_inv')
% title(t,'Singular Values');
% 
% figure;
% t=tiledlayout(1,4);
% t.TileSpacing='tight';
% title(t,'$\chi$ Reconstructions - Noisy Data','Interpreter','latex');
% 
% nexttile; %% left original chi
% imagesc(x_values, y_values, chi); 
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Original contrast ${\chi}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile; %pinv
% imagesc(x_values, y_values, abs(xmn_noisy));
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - Pinv','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile; %svd
% imagesc(x_values, y_values, abs(xmn_svd_noisy));
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - SVD','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile;% truncated svd
% imagesc(x_values, y_values, abs(xmn_trunc_svd_noisy));
% colormap(flipud('Gray'))
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - Truncated SVD','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% outerPos = t.OuterPosition;
% outerPos(2) = outerPos(2) + .25;
% outerPos(4) = outerPos(4) - .4; 
% t.OuterPosition = outerPos;
% 
% saveas(gcf,'noisy_reconstructions_square_setup', 'epsc');

% %% CIRCLE SETUP
% 
% lambda=lambda_source;
% x_rec=[];
% y_rec=[];
% 
% % Center of the object domain
% center_x = (x0 + x1) / 2;
% center_y = (y0 + y1) / 2;
% 
% % Define the radius of the circle (choose a reasonable size, e.g., 1.5λ)
% radius = 2 * lambda_source;
% 
% % Define the number of receivers
% M = 360;
% 
% 
% % Compute the angles for the receiver positions
% theta = linspace(0, 2*pi, M + 1);
% theta(end) = [];  % Remove the last point to avoid overlap with the first
% 
% % Compute the receiver positions on the circle
% x_rec = center_x + radius * cos(theta);
% y_rec = center_y + radius * sin(theta);
% 
% % Create a figure
% figure(1);
% hold on;
% 
% % Plot the object domain
% rectangle('Position', [x0, y0, lambda_source, lambda_source], 'EdgeColor', 'b', 'LineWidth', 2);
% 
% % Plot the circle
% rectangle('Position', [center_x - radius, center_y - radius, 2*radius, 2*radius], ...
%           'Curvature', [1, 1], 'EdgeColor', 'r', 'LineWidth', 2);
% 
% % Plot the receiver positions
% plot(x_rec, y_rec, 'ro-', 'LineWidth', 1, 'MarkerSize', 4);
% 
% % Set axis properties
% axis equal tight;
% xlim([x0 - 2*lambda, x1 + 2*lambda]);
% ylim([y0 - 2*lambda, y1 + 2*lambda]);
% title('Receiver Positions on a Circle Centered Around the Object Domain')
% xlabel('x-axis (horizontal)');
% ylabel('y-axis (vertical)');
% hold off;
%
% saveas(gcf,'setup_circle','epsc');
% 
% %%% 
% lambda=2*pi/kb;
% freq=c/lambda;
% 
% A=zeros(M,N);
% receivers_locs=[x_rec; y_rec];
% 
% G=@(r) -1j/4 * besselh(0,2,r); % define green's func
% 
% for i=1:M
%     dist_prime = sqrt((receivers_locs(1,i) - X).^2 + (receivers_locs(2,i)-Y).^2);
%     for m=1:Ny
%         for n=1:Nx
%             a_temp(m,n)=G(kb*dist_prime(m,n))*uinc(m,n);
%         end
%     end
%     A(i,:)=reshape(a_temp,1,[]);  
% end
% 
% chi_flat=reshape(chi,[size(chi,1)*size(chi,2) 1]);
% 
% uscat=A*chi_flat.*kb^2;
% 
% %%% Contrast Reconstructions - Noiseless Cases
% xmn=pinv(A)*uscat;
% xmn=reshape(xmn,Ny,Nx);
% 
% figure;
% t=tiledlayout(1,3);
% t.TileSpacing='tight';
% title(t,'$\chi$ Reconstructions - Noiseless Cases','Interpreter','latex');
% 
% nexttile; %% left original chi
% imagesc(x_values, y_values, chi); 
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Original contrast ${\chi}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile; % middle using pinv
% imagesc(x_values, y_values, abs(xmn));
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - Pinv','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% % Pinv from SVD
% [U, S, V] = svd(A, 'econ', 'vector');
% S_inv = 1 ./(S+eps); %add eps to avoid nans or infs 
% xmn_svd = V * diag(S_inv) * U' * uscat;
% xmn_svd=reshape(xmn_svd,Ny,Nx);
% 
% nexttile;
% imagesc(x_values, y_values, abs(xmn_svd));
% colormap(flipud('Gray'))
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - SVD based','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% outerPos = t.OuterPosition;
% outerPos(2) = outerPos(2) + .25; % Adjust the bottom position
% outerPos(4) = outerPos(4) - .35; % Adjust the height
% t.OuterPosition = outerPos;
% 
% saveas(gcf,'noiseless_reconstructions_circle_setup', 'epsc');
% 
% %%% Noisy cases
% noise_std=0.1;
% uscat_noisy=uscat+randn(M,1)*noise_std;
% 
% xmn_noisy=pinv(A)*uscat_noisy;
% xmn_noisy=reshape(xmn_noisy,[Ny Nx]);
% 
% % Pinv from SVD
% [U, S, V] = svd(A, 'econ', 'vector');
% S_inv = 1 ./(S+eps); %add eps to avoid nans or infs 
% 
% xmn_svd_noisy= V * diag(S_inv) * U' * uscat;
% xmn_svd_noisy=reshape(xmn_svd_noisy,[Ny Nx]);
% 
% truncSVD = @(U,S,V,p) V(:,1:p)*diag(S(1:p))*U(:,1:p)';
% 
% k=300;
% xmn_trunc_svd_noisy=truncSVD(U,S_inv,V,k)*uscat;
% xmn_trunc_svd_noisy=reshape(xmn_trunc_svd_noisy,[Ny Nx]);
% 
% figure(33)
% t=tiledlayout(1,2);
% nexttile; semilogy(S); subtitle('S');
% nexttile; semilogy(S_inv); subtitle('S_inv')
% title(t,'Singular Values');
% 
% figure;
% t=tiledlayout(1,4);
% t.TileSpacing='tight';
% title(t,'$\chi$ Reconstructions - Noisy Data','Interpreter','latex');
% 
% nexttile; %% left original chi
% imagesc(x_values, y_values, chi); 
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('Original contrast ${\chi}$','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile; %pinv
% imagesc(x_values, y_values, abs(xmn_noisy));
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - Pinv','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile; %svd
% imagesc(x_values, y_values, abs(xmn_svd_noisy));
% colormap(flipud('Gray')); C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - SVD','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% nexttile;% truncated svd
% imagesc(x_values, y_values, abs(xmn_trunc_svd_noisy));
% colormap(flipud('Gray'))
% C = colorbar('location','southoutside');
% axis equal tight;
% subtitle('${x}_{mn}$ - Truncated SVD','Interpreter','latex');
% xlabel('x');
% ylabel('y');
% 
% outerPos = t.OuterPosition;
% outerPos(2) = outerPos(2) + .25; % Adjust the bottom position
% outerPos(4) = outerPos(4) - .4; % Adjust the height
% t.OuterPosition = outerPos;
% 
% saveas(gcf,'noisy_reconstructions_circle_setup', 'epsc');


