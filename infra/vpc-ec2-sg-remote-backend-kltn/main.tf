provider "aws" {
  region = "us-east-1"

}


terraform {
  

  backend "s3" {
  bucket         = "tranchucthien-s3"
  key            = "spark_env/terraform.tfstate"
  region         = "us-east-1"
  use_lockfile   = true
  encrypt = true
}

  required_version = ">= 1.0.0"
}
# module "vpc" {
#   source = "./modules/vpc"
#   name   = "tct-vpc"

# }

module "ec2" {
  source   = "./modules/ec2-sg" # Path to the module directory
  key_name = "emr"
  ec2_name = "spark_terra"   # Example tag name for EC2 instance, can be overridden
  key_path = "./emr.pem" # Path to the private key file
  instance_ami = "ami-03594924a9d99dce1"
  instance_type = "t2.large" # 
  ingress_rules = [
    { from_port = 22, to_port = 22, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] },   # SSH
    { from_port = 80, to_port = 80, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] },   # HTTP
    { from_port = 443, to_port = 443, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] }, # HTTPS
  ]
  sg_description = "SG with SSH and HTTP/HTTPS access"
  vpc_id         = "vpc-0e37a52f77163ad56"
  subnet_id      = "subnet-015002dd690f2feb5"
  provisioner_commands = [
    "ls"
  ]
}

