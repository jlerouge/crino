function y = imcircle(im_size, i, j, radius, contour)
    if(nargin <= 3)
        radius = 1;
    end
    if(nargin <= 4)
        contour = false;
    end
    radiusI = ceil(radius);
    box = [max(i-radiusI,1), min(i+radiusI, im_size(1)); max(j-radiusI,1), min(j+radiusI, im_size(2))];
    [Y,X] = meshgrid(box(1,1):box(1,2), box(2,1):box(2,2));
    if(contour)
        temp = -((X-j).^2+(Y-i).^2<=(radius+1)^2);
    else
        temp = 0;
    end
    temp = temp + (2^contour)*((X-j).^2+(Y-i).^2<=radius^2);
    y = zeros(im_size);
    y(box(2,1):box(2,2), box(1,1):box(1,2)) = temp;
end